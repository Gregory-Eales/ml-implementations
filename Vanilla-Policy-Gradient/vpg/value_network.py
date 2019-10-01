import torch

class ValueNetwork(torch.nn.Module):

	def __init__(self, alpha, input_dims, output_dims):

		# inherit from nn module class
		super(ValueNetwork, self)__init__()
