import torch

class PolicyNetwork(torch.nn.Module):

	def __init__(self, alpha, input_dims, output_dims):

		# inherit from nn module class
		super(PolicyNetwork, self)__init__()

	# initialize network
	def initialize_network(self):
		pass

	# define loss function
	def loss(self):
		pass

	# training loop
	def train(self, x, y, iter):
		pass

class VPG(object):

	def __init__(self, alpha, input_dims, output_dims):

		# initialize policy network
		self.policy_network = PolicyNetwork(alpha, input_dims, output_dims)

	def act(self):
		pass
