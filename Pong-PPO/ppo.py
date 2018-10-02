import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from parallelEnv import parallelEnv
import pong_utils
from torch.distributions import Categorical


class PPO:
    def __init__(self, envs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = Policy().to(self.device)
        self.envs = envs
        self.n_actions = 2

    def clipped_surrogate(self, old_probs, states, actions, rewards,
                          discount=0.995, epsilon=0.1, beta=0.01):
        discount = discount ** np.arange(len(rewards))
        rewards = np.asarray(rewards) * discount[:, np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_probs = torch.stack(old_probs).squeeze(2)

        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=self.device)

        # convert states to policy (or probability)
        new_probs = self.states_to_prob(states, actions)

        # ratio for clipping
        ratio = new_probs / old_probs

        # clipped function
        clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
                    (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + beta * entropy)

    def train(self, episode=800, discount_rate=0.99, epsilon=0.1, beta=0.01,
              tmax=320, SGD_epoch=4, lr=1e-4):
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # keep track of progress
        mean_rewards = []

        for e in tqdm(range(episode)):
            # collect trajectories
            old_probs, states, actions, rewards = \
                self.collect_trajectories(tmax=tmax)

            total_rewards = np.sum(rewards, axis=0)

            # gradient ascent step
            for _ in range(SGD_epoch):
                # uncomment to utilize your own clipped function!
                # L = -clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

                L = -self.clipped_surrogate(old_probs, states, actions, rewards,
                                            epsilon=epsilon, beta=beta)
                self.optimizer.zero_grad()
                L.backward(retain_graph=True)
                self.optimizer.step()
                del L

            # the clipping parameter reduces as time goes on
            epsilon *= .999

            # the regulation term also reduces
            # this reduces exploration in later runs
            beta *= .995

            # get the average reward of the parallel environments
            mean_rewards.append(np.mean(total_rewards))

            # display some progress every 20 iterations
            if (e + 1) % 20 == 0:
                print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
                print(total_rewards)

    def states_to_prob(self, states, actions):
        statesv = torch.stack(states)
        policy_input = statesv.view(-1, *statesv.shape[-3:])
        policy_output = self.policy(policy_input).view([320, 8, 2])  # t_max, n_workers, n_actions
        probs = torch.gather(policy_output, 2, actions).squeeze(2)
        return probs

    # collect trajectories for a parallelized parallelEnv object
    def collect_trajectories(self, tmax=200, nrand=5):

        # number of parallel instances
        n = len(self.envs.ps)

        # initialize returning lists and start the game!
        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        self.envs.reset()

        # start all parallel agents
        self.envs.step([1] * n)

        # perform nrand random steps
        for _ in range(nrand):
            fr1, re1, _, _ = self.envs.step(np.random.choice([pong_utils.RIGHT, pong_utils.LEFT], n))
            fr2, re2, _, _ = self.envs.step([0] * n)

        for t in range(tmax):

            # prepare the input
            # preprocess_batch properly converts two frames into
            # shape (n, 2, 80, 80), the proper input for the policy
            # this is required when building CNN with pytorch
            batch_input = pong_utils.preprocess_batch([fr1, fr2])

            # probs will only be used as the pi_old
            # no gradient propagation is needed
            # so we move it to the cpu
            probs_tensor = self.policy(batch_input)
            m = Categorical(probs_tensor)
            action = m.sample().unsqueeze(1)
            probs = torch.gather(probs_tensor, 1, action)
            action = action.cpu().numpy()

            # advance the game (0=no action)a
            # we take one action and skip game forward
            fr1, re1, is_done, _ = self.envs.step(action + 4)
            fr2, re2, is_done, _ = self.envs.step([0] * n)

            reward = re1 + re2

            # store the result
            state_list.append(batch_input)
            reward_list.append(reward)
            prob_list.append(probs)
            action_list.append(action)

            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if is_done.any():
                break

        # return pi_theta, states, actions, rewards, probability
        return prob_list, state_list, \
               action_list, reward_list

class Policy(nn.Module):
    def __init__(self, n_actions = 2):
        super(Policy, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)

if __name__ == '__main__':
    envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)
    agent = PPO(envs=envs)
    agent.train()
