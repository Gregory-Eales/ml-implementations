import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

# neural network class
class NeuralNetwork(object):

	def __init__(self, input_shape=1, output_shape=1, num_layers=3):

		# check for correct input types and values
		assert type(input_shape) == int, "input shape needs to be an integer"
		assert input_shape > 0, "input shape needs to be greater than 0"

		assert type(output_shape) == int, "input type needs to be an integer"
		assert output_shape > 0, "output shape needs to be greater than 0"

		assert type(num_layers) == int, "number of layers needs to be an integer"
		assert num_layers >= 3, "number of layers needs to be 3 or larger"

		

		# internalize network parameters
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

			if (self.num_layers-1) == i:
				self.w["w" + str(i)] = torch.randn(self.input_shape+1, self.output_shape, dtype=torch.float32).cuda()

			elif i == 1:
				self.w["w" + str(i)] = torch.randn(self.input_shape, self.input_shape+1, dtype=torch.float32).cuda()

			else:
				self.w["w" + str(i)] = torch.randn(self.input_shape+1, self.input_shape+1, dtype=torch.float32).cuda()


	def initialize_bias(self):

		self.b = {}

		for i in range(1, self.num_layers):

			if (self.num_layers-1) == i:
				self.b["b" + str(i)] = torch.randn(self.output_shape, dtype=torch.float32).cuda()

			else:
				self.b["b" + str(i)] = torch.randn(self.input_shape+1, dtype=torch.float32).cuda()



	def initialize_activations(self):
		self.a = {}

	def initialize_z(self):
		self.z = {}

	######################
	# Activation Methods #
	######################

	def sigmoid(self, z):
		return torch.sigmoid(z)

	def sigmoid_prime(self, z):
		sig = torch.sigmoid(z)
		return sig*(1-sig)

	def tanh(self, z):
		return torch.tanh(z)

	def tanh_prime(self, z):
		t = torch.tanh(z)
		return (1 - t**2)

	####################
	# Learning Methods #
	####################

	def predict(self, x):

		self.a["a0"] = x.cuda()
		last_layer = self.num_layers-1
		
		for i in range(1, self.num_layers-1):
			self.z["z" + str(i)] = torch.mm(self.a["a"+str(i-1)], self.w["w"+str(i)])# + self.b["b" + str(i)]
			self.a["a" + str(i)] = self.tanh(self.z["z" + str(i)])


		self.z["z" + str(last_layer)] = torch.mm(self.a["a"+str(last_layer-1)], self.w["w"+str(last_layer)]) #+ self.b["b" + str(last_layer)]
		self.a["a" + str(last_layer)] = self.sigmoid(self.z["z" + str(last_layer)])



		return self.a["a" + str(last_layer)]

	def cost(self, y):
		return torch.sum((y.cuda() - self.a["a"+str(self.num_layers-1)])**2)

	def cost_prime(self, y_hat, y):
		#return (y/self.a['a' + str(self.num_layers-1)] - (1-y)/(1-self.a['a' + str(self.num_layers-1)]))
		return y_hat - y.cuda()

	def calculate_updates(self, y, alpha):
		cost_prime = self.cost_prime(self.a["a"+str(self.num_layers-1)], y)

		for i in reversed(range(1, self.num_layers)):

			if i != (self.num_layers-1):
				w_curr = self.w["w"+str(i)]
				w_prev = self.w["w"+str(i+1)]
				a_prime = self.tanh_prime(self.z["z"+str(i)])
				a_ahead = self.a["a"+str(i-1)]
				prev_update = self.updates["w" + str(i+1)]
				self.updates["w"+str(i)] = torch.mm(prev_update, torch.t(w_prev)) * a_prime
				self.b["b"+ str(i)] -= self.updates["w"+str(i)].sum(0)*alpha/self.a["a0"].shape[0]

			else:
				self.updates["w"+str(i)] = (cost_prime * self.sigmoid_prime(self.z["z"+str(i)]))
				self.b["b"+ str(i)] -= self.updates["w"+str(i)].sum(0)*alpha/self.a["a0"].shape[0]

		for i in reversed(range(1, self.num_layers)):
			a_ahead = self.a["a"+str(i-1)]
			self.updates["w"+str(i)] = torch.mm(torch.t(a_ahead), self.updates["w"+str(i)])

	def update_weights(self, alpha):
		for i in range(1, self.num_layers):
			self.w["w" + str(i)] -= alpha*self.updates["w" + str(i)]/self.a["a0"].shape[0]

	def train(self, x, y, batch_size=10, alpha=0.1, iterations=10):
		self.historical_cost = []
		for iter in tqdm(range(iterations)):
			# make prediction
			self.predict(x)
			# calculate error
			cost = self.cost(y)/self.a["a"+str(self.num_layers-1)].shape[0]
			# record error
			self.historical_cost.append(cost)
			# calculate update
			self.calculate_updates(y, alpha)
			# update weights
			self.update_weights(alpha)


def main():

	# create data
	x = torch.ones(10, 5)
	y = torch.ones(10, 2)

	# make prediction
	NN = NeuralNetwork(5, 2, 8)
	NN.train(x, y, iterations=100, alpha=0.01)
	plt.plot(NN.historical_cost)
	plt.show()

if __name__ == "__main__":
	main()
