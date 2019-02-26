import numpy as np 
import tensorflow as tf
from tqdm import tqdm
import time


# TO DO:
# 	- create weights based on data, and topology
# 	- pick activation function for each layer
# 	- make forward function
#	- make activation functions
# 	- make derivative activation functions
# 	- make update function
# 	- make training function



class NeuralNetwork(object):


	def __init__(self, num_layers=3, input_shape=[1, 1]):

		self.w = {}
		self.b = {}
		self.a = {}
		self.z = {}
		self.num_layers = num_layers
		self.create_weights()


	def create_weights(self, num_layers=3, input_shape=[1, 1], output_shape=[1, 1]):

		# make sure there are three or more layers
		if self.num_layers < 3:
			raise Exception("num_layers should not be less than 3, value given was: " + str(num_layers))
		
			# create first set
			self.w["w1"] = np.random.rand(input_shape[1]+1, input_shape[1])
			self.b["b1"] = np.random.rand(input_shape[1]+1, 1)

			# create hidden layers
			for i in range(2, self.num_layers):

				self.w["w" + str(i)] = np.random.rand(input_shape[1]+1, input_shape[1]+1)
				self.b["b" + str(i)] = np.random.rand(input_shape[1]+1, 1)

			# create last layer
			self.w["w" + str(self.num_layers)] = np.random.rand(output_shape[1], input_shape[1]+1)
			self.b["b" + str(self.num_layers)] = np.random.rand(output_shape[1], 1)


	def train(self, x, y, iterations=1, alpha=0.01):

		# loop through data set each iteration
		print("    Training NeuralNetwork")
		for i in tqdm(range(iterations)):
			time.sleep(0.1)


	def predict(self, x):

		# multiply then activate first layer
		self.z["z1"] = self.dot(self.w["w1"], x.T) + self.b["b1"]
		self.a["a1"] = self.tanh(self.z["z1"])

		# loop through hidden layers, multiply then activate
		for i in range(2, self.num_layers):
			self.z["z" + str(i)] = self.dot(self.w["w"+str(i)], self.a["a" + str(i-1)])
			self.a["a" + str(i)] = self.tanh(self.z["z" + str(i)])

		# multiply then activate last layer
		self.z["z" + str(self.num_layers)] = self.dot(self.w["w"+str(self.num_layers)], self.a["a" + str(self.num_layers-1)])
		self.a["a" + str(self.num_layers)] = self.sigmoid(self.z["z" + str(self.num_layers)])

		return self.a["a" + str(self.num_layers)]
		

	def sigmoid(self, z):
		
		# use library based on matrix size and GPU access
		if True:
			return tf.divide(1.000, tf.add(1.000, tf.exp(-z)))

		else:
			return 1.000 / (1.000 + np.exp(-z))

	def sigmoid_prime(self, z):

		if True:
			return tf.multiply(self.sigmoid(z), (tf.sub(1, self.sigmoid(z))))

		else:
			return self.sigmoid(z)*(1-self.sigmoid(z))


	def tanh(self, z):

		# use library based on matrix size and GPU access
		if True:
			return tf.tanh(z)

		else:
			return np.tanh(z)


	def tanh_prime(self, z):

		# use library based on matrix size and GPU access
		if True:
			return 1 - tf.square(tf.tanh(z))

		else:
			return 1 - np.square(np.tanh(z))

	def relu(self, z):

		if True:
			return np.where(z > 0, 1, 0)

		else:
			None



	self dot(self, a, b):

		if True:
			return tf.matmul(a, b)

		else:
			np.dot(a, b)











