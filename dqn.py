import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import deque
import random

class Net(nn.Module):
    def __init__(self, n_states, hidden_dim, actions_dim):
        super(Net, self).__init__()
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.action_dim = actions_dim
        self.fc1 = nn.Linear(self.n_states, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.action_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen = self.capacity)

    def add(self, state, act, reward, state_):
        one_try = (state, act, reward, state_)
        self.buffer.append(one_try)

    def sample(self, sample_size):
        batch = random.sample(self.buffer, sample_size)
        states, acts, rewards, state_s = zip(*batch)
        return np.array(states), acts, rewards, np.array(state_s)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

class DQN(nn.Module):
    def __init__(self, states_nums, hidden_dim, action_nums, lr, gamma, epsilon, target_update_nums, capacity, sample_size):
        super(DQN, self).__init__()
        self.states_nums = states_nums
        self.hidden_dim = hidden_dim
        self.action_nums = action_nums
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.target_update_nums = target_update_nums
        self.eval_net, self.target_net = Net(self.states_nums, self.hidden_dim, self.action_nums).cuda(), Net(self.states_nums, self.hidden_dim, self.action_nums).cuda()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = self.lr)
        self.loss_func = torch.nn.MSELoss()
        self.memery = ReplayBuffer(capacity)
        self.sample_size = sample_size
        self.count = 0

    def choose_action(self, x):
        if np.random.uniform() < self.epsilon:
            with torch.no_grad():
                action_value = self.eval_net(x).tolist()[0]
                action = action_value.index(max(action_value))
        else:
            action = np.random.choice(2)
        return action
    def store_transition(self, state, act, reward, state_):
        self.memery.add(state, act, reward, state_)

    def update(self):
        states, acts, rewards, states_ = self.memery.sample(sample_size=self.sample_size)
        states, states_ = torch.tensor(list(states), dtype=torch.float32).cuda(), torch.tensor(list(states_), dtype=torch.float32).cuda()
        rewards = list(rewards)
        rewards = torch.tensor(rewards, dtype=torch.float32).cuda()
        q_values = self.eval_net(states).gather(1, torch.tensor(acts).view(-1, 1).cuda())
        q_next = self.target_net(states_).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * q_next
        dqn_loss = torch.mean(self.loss_func(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update_nums == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.count += 1