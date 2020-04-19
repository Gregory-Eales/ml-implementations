import torch
import numpy


class Buffer(object):

    def __init__(self):
        
        self.reset()

    def reset(self):
    	self.state_buffer = []
        self.reward_buffer = []
        self.discount_reward_buffer = []
        self.action_buffer = []
        self.policy_buffer = []
        self.advantage_buffer = []


    def store_state(self, s):
    	pass

    def store_reward(self, r):
    	pass

    def store_disc_reward(self, disc_r):
    	pass

    def store_action(self, act):
    	pass

    def store_policy(self, p):
    	pass

    def store_advantage(self, adv):
    	pass

    def get_state(self, s):
    	pass

    def get_reward(self, r):
    	pass

    def get_disc_reward(self, disc_r):
    	pass

    def get_action(self, act):
    	pass

    def get_policy(self, p):
    	pass

    def get_advantage(self, adv):
    	pass


