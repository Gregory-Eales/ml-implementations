import torch
import numpy as np


class Buffer(object):

    def __init__(self):

        self.policy_buffer = []
        self.old_policy_buffer = []
        self.advantage_buffer = []
        self.state_buffer = []


    def store_advantages(self, adv):
        self.advantage_buffer.append(adv)

    def store_policy(self, p):
        self.policy_buffer.append(p)

    def store_old_policy(self, old_p):
        self.old_policy_buffer.append(old_p)

    def store_state(self, state):
        self.state_buffer.append(state)


    def store(self, state, policy, old_policy):
        self.store_state(state)
        self.store_old_policy(old_policy)
        self.store_policy(policy)

    def get_old_policy(self):
        pass

    def get_policy(self):
        pass

    def get_states(self):
        pass

    def get_advantages(self):
        pass

    def get(self):

        return
