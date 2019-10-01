import torch

class PolicyNetwork(torch.nn.Module):

	def __init__(self, alpha, input_dims, output_dims):

		# inherit from nn module class
		super(PolicyNetwork, self)__init__()

		# initialize network
		self.initialize_network()

	# initialize network
	def initialize_network(self):

		# define network components
		self.batch_norm1 = torch.nn.BatchNorm1d()
		self.fc1 = torch.nn.Linear(4)
		self.relu1 = torch.nn.ReLU()

	def predict(x):
		out = self.batch_norm(x)
		out = self.fc1(out)

	# define loss function
	def loss(self):
		pass

	# training loop
	def train(self, x, y, iter):
		pass
