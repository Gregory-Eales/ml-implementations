import torch
import numpy


class Buffer(object):

	def __init__(self):
		
		self.reset()

	def reset(self):
		self.observation_buffer = []
		self.reward_buffer = []
		self.discount_reward_buffer = []
		self.action_buffer = []
		self.policy_buffer = []
		self.advantage_buffer = []
		self.terminal_buffer = []


	def store_terminal(self, done):
		if done: self.terminal_buffer.append(1)
		else: 0


	def store(self, observation, reward, done):
		self.store_observation(observation)
		self.store_reward(reward)

	def store_observation(self, observation):
		self.observation_buffer.append(observation)

	def store_reward(self, r):
		self.reward_buffer.append(r)

	def store_disc_reward(self, disc_r):
		self.discount_reward_buffer += disc_r

	def store_action(self, act):
		self.action_buffer.append(act)

	def store_policy(self, p):
		self.policy_buffer.append(p)

	def store_advantage(self, adv):
		self.advantage_buffer.append(adv)

	def get_observation(self):
		return self.observation_buffer

	def get_reward(self):
		return self.reward_buffer

	def get_disc_reward(self):
		return self.discount_reward_buffer

	def get_action(self):
		return self.action_buffer

	def get_policy(self):
		return self.policy_buffer

	def get_advantage(self):
		return self.advantage_buffer


