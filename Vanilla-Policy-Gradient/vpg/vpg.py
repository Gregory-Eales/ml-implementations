import torch

class Buffer(object):

	def __init__(self):
		pass

class VPG(object):

	def __init__(self, alpha, input_dims, output_dims):

		# initialize policy network
		self.policy_network = PolicyNetwork(alpha, input_dims, output_dims)

		self.value_network = ValueNetwork(alpha, input_dims, output_dims)

		self.vpg_buffer = Buffer()

	def act(self):
		pass
