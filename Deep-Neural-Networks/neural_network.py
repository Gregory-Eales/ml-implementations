import numpy
import torch




# neural network class
class NeuralNetwork(object):

	def __init__(self, input_shape, output_shape, num_layers):
		

		# internalize network parameters
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.num_layers = num_layers

		# initialize weights
		self.w = self.initialize_weights()

		# initialize bias cache
		self.b = self.initialize_bias()

		# initialize activation caches
		self.a = self.initialize_activations()
		pass

		# initialize z caches
		self.z = self.initialize_z()

	def initialize_weights(self):
		pass

	def initialize_bias(self):
		pass

	def initialize_activations(self):
		pass

	def initialize_z(self):
		pass






