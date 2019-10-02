import logging

import torch

from buffer import Buffer
from value_network import ValueNetwork
from policy_network import PolicyNetwork

class VPG(object):

	def __init__(self, alpha, input_dims, output_dims):

		# initialize policy network
		self.policy_network = PolicyNetwork(alpha, input_dims, output_dims)

		# initialize value network
		self.value_network = ValueNetwork(alpha, input_dims, output_dims)

		# initialize vpg buffer
		self.vpg_buffer = Buffer()

	def act(self, s):
		return self.policy_network(s)

	def update(self, iterations):
		pass

	def playthrough(self, env):
		pass

	def train(self, num_episodes, env):

		# for each iteration:
		for episode in range(num_episodes):

		# playthrough an episode to obtain trajectories T
			self.playthrough(env)

		# compute "rewards-to-go"

		# compute advantage estimates

		# estimate policy gradient

		# compute policy gradient

		# update policy
		pass
