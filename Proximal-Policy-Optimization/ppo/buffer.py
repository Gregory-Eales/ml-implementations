import torch
import numpy as np


class Buffer(object):

    def __init__(self):

        self.policy_buffer = []
        self.old_policy_buffer = []
        self.advantage_buffer = []
        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []


    def store_advantages(self, adv):
        self.advantage_buffer.append(adv)

    def store_policy(self, p):
        self.policy_buffer.append(p)

    def store_old_policy(self, old_p):
        self.old_policy_buffer.append(old_p)

    def store_state(self, state):
        self.state_buffer.append(state)

    def store_trajectory(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def store(self, state, policy, old_policy):
        self.store_state(state)
        self.store_old_policy(old_policy)
        self.store_policy(policy)

    def get_old_policy(self):
        return self.old_policy_buffer

    def get_policy(self):
        pass

    def get_states(self):
        return self.state_buffer

    def get_advantages(self):
        return self.advantage_buffer

    def get_rewards(self):
        return self.reward_buffer

    def get(self):

        return
