import numpy as np
import pytorch as pt


class CNN(object):

	def __init__(self, num_convs, num_dense, dense_output):
		
		# define network topology
		self.num_convs = num_convs
		self.num_dense = num_dense
		self.dense_output = dense_output

		# define weights
		self.dense_w = {}
		self.pool_w = {}
		self.conv_w {}

		# initiate weights
		self.initiate_weights()




	def initiate_weights(self):
		

		# initiating dense weights
		for i in range(1, num_dense):
			self.dense_w["w" + str(i)] = np.random.rand(self.dense_output+1, self.dense_output+1)

		self.dense_w["w" + str(self.)] = np.random.rand(self.dense_output, self.dense_output+1)



	def predict(self):
		pass

	def train(self):
		pass

	def single_conv(self):
		pass

	def conv_forward(self):
		pass

	def single_pool(self):
		pass

	def pad(self):
		pass

	def sigmoid(self):
		pass

	def sigmoid_prime(self):
		pass

	def tanh(self):
		pass

	def tanh_prime(self):
		pass

	def dense_forward(self):
		pass