import torch


class Buffer(object):

    def __init__(self):

        # store actions
        self.action_buffer = []

        # store state
        self.observation_buffer = []

        # store reward
        self.reward_buffer = []

    def store_obs(self, obs):
        self.observation_buffer.append(obs)

    def store_reward(self, reward):
        self.reward_buffer.append(reward)

    def store_actions(self, action):
        self.action_buffer.append(action)
