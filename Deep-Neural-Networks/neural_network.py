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


	def __init__(self):
		pass

	def sigmoid(self, z):
		
		# use library based on matrix size
		if True:
			return tf.divide(1.000, tf.add(1.000, tf.exp(-z)))

		else:
			return 1.000 / (1.000 + np.exp(-z))

	def tanh(self, z):

		# use library based on matrix size
		if True:
			return tf.tanh(z)

		else:
			return np.tanh(z)

