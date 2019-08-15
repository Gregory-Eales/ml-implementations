import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

# neural network class
class NeuralNetwork(object):

	def __init__(self, input_shape=1, output_shape=1, num_layers=3, gpu=False):

		# check for correct input types and values
		assert type(input_shape) == int, "input shape needs to be an integer"
		assert input_shape > 0, "input shape needs to be greater than 0"

		assert type(output_shape) == int, "input type needs to be an integer"
		assert output_shape > 0, "output shape needs to be greater than 0"

		assert type(num_layers) == int, "number of layers needs to be an integer"
		assert num_layers >= 3, "number of layers needs to be 3 or larger"

		assert type(gpu) == bool, "type of gpu needs to be bool"

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
		self.updates = {}

		self.historical_cost = []

		# initialize weight cache
		self.initialize_weights()
		# initialize bias cache
		self.initialize_bias()
		# initialize activation cache
		self.initialize_activations()
		# initialize z cache
		self.initialize_z()

	##########################
	# Initialization Methods #
	##########################

	def initialize_weights(self):

		self.w = {}

		for i in range(1, self.num_layers):

			# use pytorch if gpu is true
			if self.gpu == True:

				if (self.num_layers-1) == i:
					self.w["w" + str(i)] = torch.randn(self.input_shape+1, self.output_shape, dtype=torch.float32)

				elif i == 1:
					self.w["w" + str(i)] = torch.randn(self.input_shape, self.input_shape+1, dtype=torch.float32)

				else:
					self.w["w" + str(i)] = torch.randn(self.input_shape+1, self.input_shape+1, dtype=torch.float32)

			# use numpy if gpu is false
			if self.gpu == False:

				if (self.num_layers-1) == i:
					self.w["w" + str(i)] = np.random.random([self.input_shape+1, self.output_shape])
					self.w["w" + str(i)] = self.w["w" + str(i)].astype('float16')

				elif (self.num_layers-1) != 1:
					self.w["w" + str(i)] = np.random.random([self.input_shape, self.input_shape+1])
					self.w["w" + str(i)] = self.w["w" + str(i)].astype('float16')

				else:
					self.w["w" + str(i)] = np.random.random([self.input_shape+1, self.input_shape+1])
					self.w["w" + str(i)] = self.w["w" + str(i)].astype('float16')

	def initialize_bias(self):

		self.b = {}

		for i in range(1, self.num_layers):

			# use pytorch if gpu is true
			if self.gpu == True:

				if (self.num_layers-1) == i:
					self.b["b" + str(i)] = torch.randn(self.output_shape, dtype=torch.float32)

				else:
					self.b["b" + str(i)] = torch.randn(self.input_shape+1, dtype=torch.float32)

			# use numpy if gpu is false
			elif self.gpu == False:

				if (self.num_layers-1) == i:
					self.b["b" + str(i)] = np.random.random([1, self.output_shape])
					self.b["b" + str(i)] = self.b["b" + str(i)].astype('float16')


				else:
					self.b["b" + str(i)] = np.random.random([1, self.input_shape+1])
					self.b["b" + str(i)] = self.b["b" + str(i)].astype('float16')

	def initialize_activations(self):

		self.a = {}

		for i in range(1, self.num_layers):

			# use pytorch if gpu is true
			if self.gpu == True:
				self.a["a" + str(i)] = torch.randn(1, dtype=torch.float32)

			# use numpy if gpu is false
			if self.gpu == False:
				self.a["a" + str(i)] = np.random.random(size=None)

	def initialize_z(self):

		self.z = {}

		for i in range(1, self.num_layers):

			# use pytorch if gpu is true
			if self.gpu == True:
				self.z["z" + str(i)] = torch.randn(1, dtype=torch.float32)

			# use numpy if gpu is false
			if self.gpu == False:
				self.w["w" + str(i)] = np.random.random(size=None)

	######################
	# Activation Methods #
	######################

	# numpy activation functions

	def sigmoid_np(self, z):
		return 1/(1+np.exp(-z))

	def sigmoid_prime_np(self, z):
		sig = sigmoid_np(z)
		return sig*(1-sig)

	def tanh_np(self, z):
		return np.tanh(z)

	def tanh_prime_np(self, z):
		return 1-np.square(np.tanh(z))

	# pytorch activation functions

	def sigmoid_torch(self, z):
		return torch.sigmoid(z)

	def sigmoid_prime_torch(self, z):
		sig = self.sigmoid_torch(z)
		return sig*(1-sig)

	def tanh_torch(self, z):
		return torch.tanh(z)

	def tanh_prime_torch(self, z):
		t = self.tanh_torch(z)
		return 1 - t**2

	####################
	# Learning Methods #
	####################

	def predict(self, x):

		self.a["a0"] = x
		last_layer = self.num_layers-1

		if self.gpu == True:

			for i in range(1, self.num_layers-1):
				self.z["z" + str(i)] = torch.mm(self.a["a"+str(i-1)], self.w["w"+str(i)]) + self.b["b" + str(i)]
				self.a["a" + str(i)] = self.tanh_torch(self.z["z" + str(i)])

			print("a: ", self.a["a"+str(last_layer-1)].shape)
			print("w: ", self.w["w"+str(last_layer)].shape)
			print("b: ", self.b["b" + str(last_layer)].shape)

			self.z["z" + str(last_layer)] = torch.mm(self.a["a"+str(last_layer-1)], self.w["w"+str(last_layer)]) + self.b["b" + str(last_layer)]
			self.a["a" + str(last_layer)] = self.sigmoid_torch(self.z["z" + str(last_layer)])

		if self.gpu == False:

			for i in range(1, self.num_layers-1):
				self.z["z" + str(i)] = np.sum(self.a["a"+str(i-1)] * self.w["w"+str(i)], axis=1) + self.b["b" + str(i)]
				self.a["a" + str(i)] = self.tanh_np(self.z["z" + str(i)])

			self.z["z" + str(last_layer)] = np.sum(self.a["a"+str(last_layer-1)] * self.w["w"+str(last_layer)], axis=1) + self.b["b" + str(last_layer)]
			self.a["a" + str(last_layer)] = self.sigmoid_np(self.z["z" + str(last_layer)])

	def cost(self, y_hat, y):
		return 0.5*(y-y_hat)**2

	def cost_prime(self, y_hat, y):
		return (y-y_hat)

	def calculate_updates(self, y):
		cost_prime = self.cost(self.a["a"+str(self.num_layers-1)], y)
		if self.gpu == True:

			for i in reversed(range(1, self.num_layers)):

				if i != (self.num_layers-1):
					w_curr = self.w["w"+str(i)]
					w_prev = self.w["w"+str(i+1)]
					a_prime = torch.reshape(torch.sum( self.tanh_prime_torch(self.z["z"+str(i)]), dim=0), [-1, 1])
					a_ahead = torch.reshape(torch.sum(self.a["a"+str(i-1)], dim=0), [-1, 1])
					#a_ahead = self.a["a"+str(i-1)]
					prev_update = self.updates["w" + str(i+1)]
					if True:
						print("This is it boys")
						print("####################")
						print("prev update: ", prev_update.shape)
						print("a prime: ", a_prime.shape)
						print("a ahead: ", a_ahead.shape)
						print("w prev: ", w_prev.shape)
						print("w current: ", w_curr.shape)
						print("#####################")


					self.updates["w"+str(i)] = torch.mm(prev_update, torch.t(w_prev))*a_prime

				else:
					self.updates["w"+str(i)] = torch.mm(torch.t(self.a["a"+ str(i-1)]), (cost_prime * self.sigmoid_prime_torch(self.z["z"+str(i)])))

			for i in reversed(range(1, self.num_layers)):
				self.updates["w"+str(i)] = torch.mm(prev_update, self.updates["w"+str(i)])

		if self.gpu == False:
			for i in reversed(range(1, self.num_layers)):

				if i != (self.num_layers-1):
					pass

				else:
					self.a["a"+str(i)] = None

	def update_weights(self, alpha):

		for i in range(1, self.num_layers):
			print("#########################")
			print("layer: ", i)
			print(self.w["w" + str(i)].shape)
			print(self.updates["w" + str(i)].shape)
			print("#########################")
			self.w["w" + str(i)] -= alpha*self.updates["w" + str(i)]



	def train(self, x, y, batch_size=10, alpha=0.1, iterations=10):

		self.historical_cost = []

		if self.gpu==True:
			for iter in tqdm(range(iterations)):

				# make prediction
				self.predict(x)
				# calculate error
				cost = self.cost(self.a["a"+str(self.num_layers-1)], y)
				# record error
				self.historical_cost.append(cost)
				# calculate update
				self.calculate_updates(y)
				# update weights
				self.update_weights(alpha)


		if self.gpu==False:
			for iter in tqdm(range(iterations)):

				# make prediction

				# calculate error
				self.historical_cost.append(self.cost())
				# calculate update

				# update weights

def main():

	# create data
	x = torch.randn(100, 5)
	y = torch.randn(100, 2)

	# make prediction
	NN = NeuralNetwork(5, 2, 4, gpu=True)
	NN.train(x, y)

if __name__ == "__main__":
	main()
