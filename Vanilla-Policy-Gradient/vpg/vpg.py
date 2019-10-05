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

		# randomly select move based on distribution
		action = np.random.choice(list(range(2)), p=action/np.sum(action))
		return action, torch.clone(prediction)


	def playthrough(self, env, n_steps, render=False):

		# create episode buffers
		reward_buffer = []
		prediction_buffer = []
		observation_buffer = []

		# reset environment
		observation = env.reset()

		for t in range(n_steps):
			# render env screen
			if render: env.render()

	        # get action, and network policy prediction
			action, prediction = self.act(observation)

			# save prediction
			prediction_buffer.append(prediction)

	        # get state + reward
			observation, reward, done, info = env.step(action)

			# save observation
			observation_buffer.append(observation)

			# update cumulative reward
			cum_reward = t+1

			# save reward
			reward_buffer.append([cum_reward])

	        # check if episode is terminal
			if done:
				reward_buffer[-1][0]=0
				# print time step
				#print("Episode finished after {} timesteps".format(t+1))

				# return values
				return reward_buffer, torch.cat(prediction_buffer), observation_buffer

	def store_data(self, rewards, actions, observations):


		observations = np.array(observations)
		rewards = np.array(rewards)

		rewards = torch.from_numpy(rewards)
		observations = torch.from_numpy(observations)

		self.vpg_buffer.reward_buffer.append(rewards)
		self.vpg_buffer.action_buffer.append(actions)
		self.vpg_buffer.observation_buffer.append(observations)

		return rewards, actions, observations

	def train(self, env, num_episodes, n_steps=10, render=False):

		for i in range(100):
			# for each iteration:
			for episode in range(num_episodes):

				# playthrough an episode to obtain trajectories T
				try:reward_buffer, action_buffer, observation_buffer = self.playthrough(env, n_steps=n_steps, render=render)
				except: pass
				# store data
				self.store_data(reward_buffer, action_buffer, observation_buffer)

				# update networks
			self.update(3)

		env.close()

	def calculate_advantages(self, observations):

		# compute state values
		v = self.value_network.predict(observations)

		# compute action function values
		q = self.act

		return

	def update(self, ):


		# calulcate rewards to go
		rtg = self.calculate_rtg()

		# calculate advantages
		advantages = self.calculate_advantages(observations)

		# update policy
		self.policy_network.update(actions, observations, rewards, iter=iterations)

		# update value network
		self.value_network.update(observations.float(), rewards.float(), iter=iterations)

	def main(self, env, n_episodes, n_steps, render=False):

		# for n episodes or terminal state:
		for episode in range(n_episodes):

			# reset environment
			observation = env.reset()

			for t in range(n_steps):

				# render env screen
				if render: env.render()

		        # get action, and network policy prediction
				action, prediction = self.act(observation)

		        # get state + reward
				observation, reward, done, info = env.step(action)

				# update model
				self.update(prediction, observation, reward)

		        # check if episode is terminal
				if done:
					reward_buffer[-1][0]=0
					# print time step
					#print("Episode finished after {} timesteps".format(t+1))

					# return values
					return reward_buffer, torch.cat(prediction_buffer), observation_buffer



def main():

	import gym

	# initialize environment
	env = gym.make('CartPole-v0')

	vpg = VPG(alpha=0.01, input_dims=4, output_dims=2)

	vpg.train(env, num_episodes=1000, n_steps=100, render=False)

	vpg.train(env, num_episodes=1, n_steps=100, render=True)

if __name__ == "__main__":
	main()
