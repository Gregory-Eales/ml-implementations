import logging

import torch
import numpy as np

from buffer import Buffer
from value_network import ValueNetwork
from policy_network import PolicyNetwork

class VPG(object):

	def __init__(self, alpha, input_dims, output_dims):

		#torch.manual_seed(1)
		#np.random.seed(1)

		# store parameters
		self.alpha = alpha
		self.input_dims = input_dims
		self.output_dims = output_dims

		# initialize policy network
		self.policy_network = PolicyNetwork(0.01, input_dims, output_dims)

		# initialize value network
		self.value_network = ValueNetwork(0.001, input_dims, output_dims)

		# initialize vpg buffer
		self.buffer = Buffer()

		# historical episode length
		self.hist_length = []

	def act(self, s):

		# convert to torch tensor
		s = torch.tensor(s).reshape(-1, len(s)).float()

		# get policy prob distrabution
		prediction = self.policy_network.forward(s)

		# convert to numpy array
		action = np.log(prediction.detach().numpy()[0])

		# randomly select move based on distribution
		action = np.random.choice(list(range(2)), p=action/np.sum(action))
		return action, torch.clone(prediction)

	def calculate_advantages(self, observation, prev_observation):

		observation = torch.from_numpy(observation).float()
		prev_observation = torch.from_numpy(prev_observation).float()

		# compute state value
		v = self.value_network.forward(prev_observation)

		# compute action function value
		q = self.value_network.forward(observation)

		# calculate advantage
		a = q-v

		return a.detach().numpy()

	def update(self, iter=1):

		# returns buffer values as pytorch tensors
		observations, actions, rewards, advantages = self.buffer.get_tensors()

		# update policy
		self.policy_network.update(actions, rewards, iter=iter)

		# update value network
		self.value_network.update(observations, rewards, iter=iter)


	def train(self, env, n_epoch, n_steps, render=False, verbos=True):

		# initialize step variable
		step = 0

		# historical episode length
		episode_lengths = [1]

		# for n episodes or terminal state:
		for epoch in range(n_epoch):

			# initial reset of environment
			observation = env.reset()

			# store observation
			self.buffer.store_observation(observation)

			episode_lengths = [1]
			# for t steps:
			for t in range(n_steps):

				# increment step
				step += 1

				# render env screen
				if render: env.render()

		        # get action, and network policy prediction
				action, prediction = self.act(observation)

				# store action
				self.buffer.store_action(prediction)

		        # get state + reward
				observation, reward, done, info = env.step(action)

				# store observation
				self.buffer.store_observation(observation)

				# store rewards
				self.buffer.store_reward(reward)

				# calculate advantage
				a = self.calculate_advantages(self.buffer.observation_buffer[-1], self.buffer.observation_buffer[-2])

				# store advantage
				self.buffer.store_advantage(a)

		        # check if episode is terminal
				if done:

					for s in reversed(range(1, step+1)):

						update = 0

						for k in reversed(range(1, s+1)):
							update += self.buffer.reward_buffer[-k]*(0.99**k)

						self.buffer.reward_buffer[-s] += update/200


					# change terminal reward to zero
					self.buffer.reward_buffer[-1] = 0

					# print time step
					if verbos:
						print("Episode finished after {} timesteps".format(step+1))

					episode_lengths.append(step)

					# reset step counter
					step = 0

					# reset environment
					observation = env.reset()

			# update model
			self.update(iter=80)
			step=0
			#print(self.buffer.reward_buffer)
			self.buffer.clear_buffer()
			print("Average Episode Length: {}".format(np.sum(episode_lengths)/len(episode_lengths)))
			print("Largest Episode Length: {}".format(max(episode_lengths)))


def main():

	# import gym
	import gym

	# initialize environment
	env = gym.make('CartPole-v0')

	vpg = VPG(alpha=0.08, input_dims=4, output_dims=2)

	vpg.train(env, n_epoch=100, n_steps=400, render=False, verbos=False)

	#vpg.train(env, n_epoch=1, n_steps=80, render=True, verbos=True)

if __name__ == "__main__":
	main()
