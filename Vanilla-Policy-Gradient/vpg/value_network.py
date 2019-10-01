import torch

class ValueNetwork(torch.nn.Module):

	def __init__(self, alpha, input_dims, output_dims):

		# inherit from nn module class
		super(ValueNetwork, self)__init__()

        # define optimizer
        self.optimizer = torch.optim.SGD(lr=alpha)

        # define loss
        self.loss = torch.nn.MSELoss()
