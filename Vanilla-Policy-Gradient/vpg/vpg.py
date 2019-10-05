import logging

import torch
import numpy as np

from buffer import Buffer
from value_network import ValueNetwork
from policy_network import PolicyNetwork

class VPG(object):

	def __init__(self, alpha, input_dims, output_dims):

		# store parameters
		self.alpha = alpha
		self.input_dims = input_dims
		self.output_dims = output_dims

		# initialize policy network
		self.policy_network = PolicyNetwork(alpha, input_dims, output_dims)

		# initialize value network
		self.value_network = ValueNetwork(alpha, input_dims, output_dims)

		# initialize vpg buffer
		self.vpg_buffer = Buffer()

	def act(self, s):

		# convert to torch tensor
		s = torch.tensor(s).reshape(-1, len(s)).float()

		# get policy prob distrabution
		prediction = self.policy_network.forward(s)

		# convert to numpy array
		action = prediction.detach().numpy()[0]

		print(list(range(2)), action)
		# randomly select move based on distrabution
		action = np.random.choice(list(range(2)), p=action/np.sum(action))

		return action, torch.copy(prediction)


	def update(self, iterations):
		pass

	def playthrough(self, env, n_steps, render=False):

		# create episode buffers
		reward_buffer = []
		action_buffer = []
		observation_buffer = []

		# reset environment
		observation = env.reset()

		for t in range(n_steps):
			# render env screen
			if render: env.render()

	        # get action, and network policy prediction
			action, prediction = self.act(observation)

	        # get state + reward
			observation, reward, done, info = env.step(action)

			# update cumulative reward
			cum_reward = t+1

	        # check if episode is terminal
			if done:

				# print time step
				print("Episode finished after {} timesteps".format(t+1))

				# return values
				return cum_reward

	def train(self, env, num_episodes, n_steps):

		# for each iteration:
		for episode in range(num_episodes):

			# playthrough an episode to obtain trajectories T
			self.playthrough(env, n_steps=n_steps, render=False)

			# compute "rewards-to-go"
			# ???????????????????

			# compute advantage estimates

			# estimate policy gradient
			self.policy_network.update(self.buffer.reward_buffer)

			# compute policy gradient

			# update policy
		pass

def main():

	import gym

	# initialize environment
	env = gym.make('CartPole-v0')

	vpg = VPG(alpha=0.01, input_dims=4, output_dims=2)

	for i in range(10):
		vpg.playthrough(env=env, n_steps=100, n_observations=100, n_actions=100, render=True)

if __name__ == "__main__":
	main()
