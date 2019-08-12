import numpy as np
import torch

# neural network class
class NeuralNetwork(object):

	def __init__(self, input_shape, output_shape, num_layers, gpu=False):

		# internalize network parameters
		self.gpu = gpu
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.num_layers = num_layers

		# create variables
		self.w = None
		self.b = None
		self.z = None
		self.a = None


		# initialize weights
		self.initialize_weights()

		# initialize bias cache
		self.initialize_bias()

		# initialize activation caches
		self.initialize_activations()

		# initialize z caches
		self.initialize_z()

	def initialize_weights(self):

		self.w = {}

		for i in range(self.num_layers):

			# use pytorch if gpu is true
			if self.gpu == True:
				self.w["w" + str(i)] = torch.randn(5, 7, dtype=torch.float16)


			# use numpy if gpu is false
			if self.gpu == False:
				self.w["w" + str(i)] = np.random.random(size=None)

	def initialize_bias(self):

		self.b = {}

		for i in range(self.num_layers):

			# use pytorch if gpu is true
			if self.gpu == True:
				self.b["b" + str(i)] = torch.randn(5, 7, dtype=torch.float16)


			# use numpy if gpu is false
			if self.gpu == False:
				self.b["b" + str(i)] = np.random.random(size=None)



	def initialize_activations(self):
		self.a = {}

		for i in range(self.num_layers):

			# use pytorch if gpu is true
			if self.gpu == True:
				self.a["a" + str(i)] = torch.randn(5, 7, dtype=torch.float16)


			# use numpy if gpu is false
			if self.gpu == False:
				self.a["a" + str(i)] = np.random.random(size=None)



	def initialize_z(self):

		self.z = {}

		for i in range(self.num_layers):

			# use pytorch if gpu is true
			if self.gpu == True:
				self.z["z" + str(i)] = torch.randn(5, 7, dtype=torch.float16)

			# use numpy if gpu is false
			if self.gpu == False:
				self.w["w" + str(i)] = np.random.random(size=None)
