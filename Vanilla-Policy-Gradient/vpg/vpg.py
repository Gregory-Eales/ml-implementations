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

		# get policy prob distrobution
		prediction = self.policy_network(s)

		# randomly select move based on distrobution


	def update(self, iterations):
		pass

	def playthrough(self, env, n_steps, n_actions, n_observations, render=False):

		# create episode buffers
		reward_buffer = torch.zeros(1, n_steps, 1)
		action_buffer = torch.zeros(1, n_steps, n_actions)
		observation_buffer = torch.zeros(1, n_steps, n_observations)

		for t in range(n_steps):
			# render env screen
	        if render: env.render()

	        # get action
	        action = self.act()

	        # get state + reward
	        observation, reward, done, info = env.step(action)

	        # check if episode is terminal
	        if done:
	            print("Episode finished after {} timesteps".format(t+1))
	            break

		pass

	def train(self, num_episodes, env):

		# for each iteration:
		for episode in range(num_episodes):

			# playthrough an episode to obtain trajectories T
			self.playthrough(env)

			# compute "rewards-to-go"
			# ???????????????????

			# compute advantage estimates

			# estimate policy gradient

			# compute policy gradient

			# update policy
		pass
