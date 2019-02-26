import numpy as np 
import tensorflow as tf


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
		self.create_weights(num_layers)


	def create_weights(self, num_layers=3):

		# make sure there are three or more layers
		if num_layers < 3:
			raise Exception("num_layers should not be less than 3, value given was: ", num_layers)
		
			# create first set
			self.w["w1"] = np.random.rand()
			self.b["b1"] = np.random.rand()

			# create hidden layers
			for i in range(2, num_layers):
				self.w["w" + str(i)] = np.random.rand()
				self.b["b" + str(i)] = np.random.rand()

			# create last layer
			self.w["w" + str(self.num_layers)] = np.random.rand()
			self.b["b" + str(self.num_layers)] = np.random.rand()


	def train(self, x, y, iterations=1, alpha):

		# loop through data set each iteration
		for i in range(iterations):
			for j in range(x.shape[0]):
				pass


	def predict(self, x):
		pass
		

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















