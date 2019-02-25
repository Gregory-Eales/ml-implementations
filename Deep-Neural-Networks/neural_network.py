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

	def tf_sigmoid(self, z):
		tf.divide(1.000, tf.add(1.000, tf.exp(-z)))

	def np_sigmoid(self, z):
		np.divide(1.000, (np.add(1.000, np.exp(-z))))


	def reg_sigmoid(self, z):
		1.000 / (1.000 + np.exp(-z))