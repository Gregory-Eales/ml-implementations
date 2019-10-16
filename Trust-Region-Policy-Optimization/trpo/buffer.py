import torch
import numpy as np

class Buffer(object):

    def __init__(self):

        self.observation_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.advantage_buffer = []

    def store_reward(self, r):
        r = torch.tensor(r)
        self.reward_buffer.append(r)

    def store_observation(self, s):
        s = torch.from_numpy(s).float()
        self.observation_buffer.append(s)

    def store_advantage(self, a):
        self.advantage_buffer.append(a)

    def store_action(self, a):
        self.action_buffer.append(a)

    def clear_buffer(self):
        self.observation_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.advantage_buffer = []

    def get_tensors(self):
        observations = torch.cat(self.observation_buffer)
        actions = torch.cat(self.action_buffer)
        rewards = torch.cat(self.reward_buffer)
        advantages = torch.cat(self.advantage_buffer)
        return observations, actions, rewards, advantages
