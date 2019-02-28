import numpy as np
import torch
from tqdm import tqdm

class CNN(object):

	# whats the best way to input network topology


	def __init__(self, num_convs, num_dense, dense_output):
		
		# define network topology
		self.num_convs = num_convs
		self.num_dense = num_dense
		self.dense_output = dense_output

		# define weights
		self.dense_w = {}
		self.conv_w  = {}
		self.dense_b = {}
		self.conv_b  = {}

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

	def single_conv(self, x, conv_layer=1, activation="tanh"):
		
		if activation == "tanh":
			return torch.tanh(torch.sum(x*self.conv_w["w" + str(conv_layer)]) + self.conv_b["b" + str(conv_layer)])

		if activation == "sigmoid":
			return torch.sigmoid(torch.sum(x*self.conv_w["w" + str(conv_layer)]) + self.conv_b["b" + str(conv_layer)])


	def conv_forward(self, x, conv_layer):

		# height, width, thickness
		x_h, x_w, x_t = x.shape[0], x.shape[1], x.shape[2]
		w_h, w_w, w_t = self.conv_w[w].shape[0], self.conv_w[w].shape[0], self.conv_w[w].shape[0], 

		h_num = 
		w_num = 
		t_num = 
		

	def single_pool(self, x):
		return torch.sum(x)

	def pad(self, a, pad_num = 1):
		padding = [pad_num, pad_num, pad_num, pad_num]
		return torch.nn.functional.pad(a, pad=padding)


	def 


	
	def sigmoid_prime(self, z):
		return torch.sigmoid(z) * (1-torch.sigmoid(z))

	def tanh_prime(self, z):
		return 1 - torch.square(torch.tanh(z))

	def dense_forward(self):
		pass

