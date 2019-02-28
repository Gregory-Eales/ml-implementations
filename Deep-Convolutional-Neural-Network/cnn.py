import numpy as np
import torch
from tqdm import tqdm

class CNN(object):

	def __init__(self, num_convs, num_dense, dense_output):
		
		# define network topology
		self.num_convs = num_convs
		self.num_dense = num_dense
		self.dense_output = dense_output

		# define weights
		self.dense_w = {}
		self.pool_w = {}
		self.conv_w  = {}

		# initiate weights
		self.initiate_weights()


	def initiate_weights(self):
		
		# initiating dense weights
		for i in range(1, self.num_dense):
			self.dense_w["w" + str(i)] = torch.rand(self.dense_output+1, self.dense_output+1)

		self.dense_w["w" + str(self.num_dense)] = torch.rand(self.dense_output, self.dense_output+1)


	def predict(self, x, y):
		
		for i in range(self.num_convs):
			pass

	def train(self):
		pass

	def single_conv(self):
		pass

	def conv_forward(self):
		pass

	def single_pool(self):
		pass

	def pad(self, a, pad_num = 1):
		padding = [pad_num, pad_num, pad_num, pad_num]
		return torch.nn.functional.pad(a, pad=padding)


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

