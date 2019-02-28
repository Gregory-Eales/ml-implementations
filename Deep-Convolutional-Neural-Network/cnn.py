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

	def train(self, x, y, iterations=1, alpha=0.1):
		
		print("    Training Convolutional Neural Network")
		for i in tqdm(range(iterations)):
			pass

	def single_conv(self, x, w, activation="tanh"):
		
		if activation == "tanh":
			return torch.tanh(torch.sum(x*w))

		if activation == "sigmoid":
			return torch.sigmoid(torch.sum(x*w))


	def conv_forward(self, x, w):
		

	def single_pool(self, x):
		return torch.sum(x)

	def pad(self, a, pad_num = 1):
		padding = [pad_num, pad_num, pad_num, pad_num]
		return torch.nn.functional.pad(a, pad=padding)


	def 


	
	def sigmoid_prime(self):
		pass

	def tanh_prime(self):
		pass

	def dense_forward(self):
		pass

