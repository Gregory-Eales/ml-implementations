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

		w = "w" + str(conv_layer)
		b = "b" + str(conv_layer)
		
		if activation == "tanh":
			return torch.tanh(torch.sum(x*self.conv_w[w]) + self.conv_b[b])

		if activation == "sigmoid":
			return torch.sigmoid(torch.sum(x*self.conv_w[w]) + self.conv_b[b])


	def conv_forward(self, x, conv_layer=1, step=1, activation="tanh"):

		

		# height, width, thickness
		w = "w" + str(conv_layer)
		x_h, x_w, x_t = x.shape[0], x.shape[1], x.shape[2]
		w_h, w_w, w_t = self.conv_w[w].shape[0], self.conv_w[w].shape[0], self.conv_w[w].shape[0], 

		h_num = self.calc_num_steps(step, x_h, w_h)
		w_num = self.calc_num_steps(step, x_w, w_w)
		t_num = self.calc_num_steps(step, x_t, w_t)

		self.conv_a[a+str(conv_layer)] = torch.zeros(h_num, w_num, t_num)

		for h in range(h_num):
			for w in range(w_num):
				for t in range(t_num):

					x_slice = x.narrow(0, h*(w_h+step-1), w_h).narrow(1, w*(w_w+step-1), w_w).narrow(2, t*(w_t+step-1), w_t)
					self.conv_a[a+str(conv_layer)][h, w, t] = self.single_conv(x_slice, conv_layer=1, activation="tanh")

	def single_pool(self, x, pool_type="average"):

		if pool_type == "average":
			return torch.sum(x)/x.numel()

		if pool_type == "max":
			return x.max()

	def pool_forward(self, x, pool_layer=1, output_shape=[1, 1, 1], step=1, pool_type="average"):
		
		# height, width, thickness
		w = "w" + str(conv_layer)
		x_h, x_w, x_t = x.shape[0], x.shape[1], x.shape[2]
		h_num, w_num, t_num = output_shape[0], output_shape[1], output_shape[2]

		w_h = self.calc_window_side(x_h, h_num)
		w_w = self.calc_window_side(x_w, w_num)
		w_t = self.calc_window_side(w_t, t_num)

		self.pool_a[a+str(pool_layer)] = torch.zeros(h_num, w_num, t_num)

		for h in range(h_num):
			for w in range(w_num):
				for t in range(t_num):

					x_slice = x.narrow(0, h*(w_h+step-1), w_h).narrow(1, w*(w_w+step-1), w_w).narrow(2, t*(w_t+step-1), w_t)
					self.conv_a[a+str(conv_layer)][h, w, t] = self.single_pool(x, pool_type)


	def pad(self, a, pad_num = 1):
		padding = [pad_num, pad_num, pad_num, pad_num]
		return torch.nn.functional.pad(a, pad=padding)

	def calc_num_steps(self, step, x, w):
		return (x-w)/(step) + 1

	def calc_window_side(step, x, length):
		return  x - ( (length - 1) * step )

	
	def sigmoid_prime(self, z):
		return torch.sigmoid(z) * (1-torch.sigmoid(z))


	def tanh_prime(self, z):
		return 1 - torch.square(torch.tanh(z))

	def dense_forward(self):
		pass

