import torch
import numpy as np

class Buffer(object):

    def __init__(self):

        self.rewards = []
        self.states = []
        self.actions = []
        self.discounted_rewards = []

    def store_reward(self, reward):
        self.rewards.append(reward)

    def store_state(self, state):
        self.states.append(state)

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

    def get_values(self):
        pass
