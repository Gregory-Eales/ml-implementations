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
			prediction_buffer.append(prediction.tolist()[0])

	        # get state + reward
			observation, reward, done, info = env.step(action)

			# save observation
			observation_buffer.append(observation)

			# update cumulative reward
			cum_reward = t+1

			# save reward
			reward_buffer.append(cum_reward)

	        # check if episode is terminal
			if done:

				# print time step
				print("Episode finished after {} timesteps".format(t+1))

				# return values
				return reward_buffer, prediction_buffer, observation_buffer

	def store_data(self, rewards, actions, observations):

		actions = np.array(actions)
		observations = np.array(observations)
		rewards = np.array(rewards)

		rewards = torch.from_numpy(rewards)
		actions = torch.from_numpy(actions)
		observations = torch.from_numpy(observations)

		self.vpg_buffer.reward_buffer.append(rewards)
		self.vpg_buffer.action_buffer.append(actions)
		self.vpg_buffer.observation_buffer.append(observations)

		return rewards, actions, observations

	def train(self, env, num_episodes, n_steps, render=False):

		# for each iteration:
		for episode in range(num_episodes):

			# playthrough an episode to obtain trajectories T
			reward_buffer, action_buffer, observation_buffer = self.playthrough(env, n_steps=n_steps, render=render)

			# store data
			self.store_data(cum_reward, action_buffer, observation_buffer)

			# update networks
			self.update()

		env.close()

	def update(self, iterations):

		rewards = torch.clone(torch.cat(self.vpg_buffer.reward_buffer))
		observations = torch.clone(torch.cat(self.vpg_buffer.observation_buffer))
		actions = torch.clone(torch.cat(self.vpg_buffer.action_buffer))

		# compute "rewards-to-go"
		# ?????
		# does this mean the reward until the terminal state?

		# compute advantage estimates

		# estimate policy gradient
		self.policy_network.update(self.buffer.reward_buffer)

		# compute policy gradient

		# update policy
		self.policy_network.train(x=actions, y=rewards)

		# update value network
		self.value_network.train(x=observations, y=rewards, iter=1)

def main():

	import gym

	# initialize environment
	env = gym.make('CartPole-v0')

	vpg = VPG(alpha=0.01, input_dims=4, output_dims=2)

	for i in range(1):
		cum_reward, action_buffer, observation_buffer = vpg.playthrough(env=env, n_steps=100, render=False)
	env.close()
	rewards, actions, observations = vpg.store_data(cum_reward, action_buffer, observation_buffer)
	print(rewards.shape, actions.shape, observations.shape)

if __name__ == "__main__":
	main()
