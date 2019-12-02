import torch
import numpy as np

class Buffer(object):

    def __init__(self):

        self.rewards = []
        self.states = []
        self.actions = []
        self.discounted_rewards = []

    def store_reward(self, reward):
        self.rewards.append(reward.tolist())

    def store_state(self, state):
        # in: numpy array
        self.states.append(state.reshape([1, 3]).tolist()[0])

    def store_action(self, action):
        self.actions.append(action)

    def discount_rewards(n_steps):
        pass

    def get_rewards(self):
        return self.rewards

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_discounted_rewards(self):
        return self.discount_rewards

    def get_data(self):
        pass
