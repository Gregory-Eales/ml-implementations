import torch
import numpy as np

class Buffer(object):

    def __init__(self):

        self.observation_buffer = []
        self.action_buffer = []
        self.log_prob_buffer = []
        self.reward_buffer = []
        self.advantage_buffer = []
        self.old_parameters = None

    def store_reward(self, r):
        r = torch.tensor(r).reshape(1,1)
        self.reward_buffer.append(r)

    def store_observation(self, s):
        s = torch.from_numpy(s).float().reshape(1, 2)
        self.observation_buffer.append(s)

    def store_advantage(self, a):
        self.advantage_buffer.append(a)

    def store_action(self, a):
        self.action_buffer.append(a)

    def store_log_prob(self, pi):
        self.log_prob_buffer.append(pi.reshape(1,1))

    def store_parameters(self, params):
        self.old_parameters = params

    def get_old_params(self):
        return self.old_parameters

    def get_log_probs(self):
        return torch.cat(self.log_prob_buffer)

    def get_actions(self):
        return torch.cat(self.action_buffer)

    def get_rewards(self):
        return torch.cat(self.reward_buffer, dim=0)

    def get_observations(self):
        return torch.cat(self.observation_buffer)[:-1].reshape(-1, 2)

    def get_advantages(self):
        return torch.cat(self.advantage_buffer)

    def clear_buffer(self):
        self.observation_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.advantage_buffer = []
        self.log_prob_buffer = []

    def get_tensors(self):
        observations = torch.cat(self.observation_buffer)
        actions = torch.cat(self.action_buffer)
        rewards = torch.cat(self.reward_buffer)
        advantages = torch.cat(self.advantage_buffer)
        return observations, actions, rewards, advantages
