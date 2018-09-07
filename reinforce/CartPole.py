import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from tqdm import tqdm

env = gym.make('CartPole-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

class Net(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Net, self).__init__()
        fc = [32, 16]
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc[0])
        self.fc2 = nn.Linear(fc[0], fc[1])
        self.fc3 = nn.Linear(fc[1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        #print(state)
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x



class Policy():
    def __init__(self, s_size=4, a_size=2):
        self.net = Net(s_size, a_size, 42)

    def forward(self, state):
        return self.net(state)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def act_max(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)  # .to(device)
        probs = self.net.forward(state)
        action = np.argmax(probs.data.numpy())
        logp = torch.log(probs[0, action])
        return action, logp

env.seed(0)
np.random.seed(0)

policy = Policy()

optimizer = optim.Adam(policy.net.parameters(), lr=1e-3)


def train(n_episodes=5000, max_t=1000, gamma=1, print_every=100):
    """Implementation of hill climbing with adaptive noise scaling.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in tqdm(range(1, n_episodes + 1)):
        state = env.reset()
        saved_log_probs = []
        rewards = []
        for t in range(max_t):
            action, logp = policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            saved_log_probs.append(logp)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)

        # print(policy_loss)
        loss = torch.stack(policy_loss).sum()

        # Minimize the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break
    return scores


scores = train()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.plot(np.arange(1, len(scores)+1), scores)
#plt.ylabel('Score')
#plt.xlabel('Episode #')
#plt.show()

env = gym.make('CartPole-v0')

for i in range(10):
    print('Trial:', i)
    state = env.reset()
    for t in range(1000):
        action, logp = policy.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break

env.close()

