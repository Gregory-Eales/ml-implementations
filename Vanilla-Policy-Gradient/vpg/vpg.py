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
		self.buffer = Buffer()

	def act(self, s):

		# convert to torch tensor
		s = torch.tensor(s).reshape(-1, len(s)).float()

		# get policy prob distrabution
		prediction = self.policy_network.forward(s)

		# convert to numpy array
		action = prediction.detach().numpy()[0]

		# randomly select move based on distribution
		action = np.random.choice(list(range(2)), p=action/np.sum(action))
		return action, torch.clone(prediction)

	def calculate_advantages(self, observation, prev_observation):

		# compute state values
		v = self.value_network.predict(prev_observation)

		# compute action function values
		q = self.value_network.predict(observation)

		# calculate advantage
		a = q-v

		return a

	def update(self, observation, action, reward):

		# update policy
		self.policy_network.update(actions, observations, rewards, iter=iterations)

		# update value network
		self.value_network.update(observations.float(), rewards.float(), iter=iterations)

	def train(self, env, n_episodes, n_steps, render=False):

		# initial reset of environment
		observation = env.reset()

		self.buffer.store_obs(observation)

		# for n episodes or terminal state:
		for episode in range(n_episodes):

			# for t steps:
			for t in range(n_steps):

				# render env screen
				if render: env.render()

		        # get action, and network policy prediction
				action, prediction = self.act(observation)

		        # get state + reward
				observation, reward, done, info = env.step(action)

				# calculate advantage
				a = self.calculate_advantages()

				# store data
				self.buffer.store_obs(observation)

		        # check if episode is terminal
				if done:
					reward_buffer[-1][0]=0

					# print time step
					print("Episode finished after {} timesteps".format(t+1))

					# reset environment
					observation = env.reset()

			# update model
			self.update()



def main():

	import gym

	# initialize environment
	env = gym.make('CartPole-v0')

	vpg = VPG(alpha=0.01, input_dims=4, output_dims=2)

	vpg.train(env, num_episodes=1000, n_steps=1000, render=False)

	vpg.train(env, num_episodes=1, n_steps=100, render=True)

if __name__ == "__main__":
	main()
