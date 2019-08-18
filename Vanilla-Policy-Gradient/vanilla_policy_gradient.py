import torch

class VPG(object):

	def __init__(self, observation_size, action_size):

		# initialize buffers
		self.observation_buffer = torch.zeros(observation_size)
		self.action_buffer = torch.zeros(action_size)
		self.advantage_buffer = torch.zeros()

	def train(self):
		pass

	def act(self):
		pass
