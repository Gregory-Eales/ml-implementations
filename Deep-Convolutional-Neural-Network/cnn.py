import numpy as np
import torch
from tqdm import tqdm
import time

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)


class CNN(object):

	# Make prediction loop
	# sort out pooling
	# make training loop
	# make backpropogation method
	# add bias updates



	def __init__(self, num_convs, num_dense, output_shape=1, input_shape = [28, 28, 1]):
		
		# define network topology
		self.num_convs = num_convs
		self.num_dense = num_dense
		self.output_shape = output_shape
		self.input_shape = input_shape

		# define weights
		self.dense_w = {}
		self.conv_w  = {}
		self.dense_b = {}
		self.conv_b  = {}
		self.pool_a = {}
		self.conv_a = {}
		self.conv_z = {}
		self.dense_a = {}

		# initiate weights
		self.initiate_weights()

	def initiate_weights(self):
		
		# initiating dense weights
		for i in range(1, self.num_dense):
			self.dense_w["w" + str(i)] = torch.rand(self.output_shape+1, self.output_shape+1)

		self.dense_w["w" + str(self.num_dense)] = torch.rand(self.output_shape, self.output_shape+1)

		# initiate convolutional weights

		self.conv_w["w1"] = torch.rand([4, 4, 1, 10])
		self.conv_w["w2"] = torch.rand([4, 4, 1, 10])
		self.conv_w["w3"] = torch.rand([4, 4, 1, 10])
		self.conv_w["w4"] = torch.rand([4, 4, 1, 10])

	def predict(self, x):

		self.conv_z["z0"] = x
		
		for i in range(self.num_convs):
			
			self.conv_forward(self.conv_z["z" + str(i)], conv_layer=i+1, step=2)


	def train(self, x, y, iterations=1, alpha=0.1):
		
		print("    Training Convolutional Neural Network")
		for i in tqdm(range(iterations)):
			pass

	def single_conv(self, x, conv_layer=1):

		w = "w" + str(conv_layer)
		b = "b" + str(conv_layer)
		return torch.sum(x*self.conv_w[w])
		


	def conv_forward(self, x, conv_layer=1, step=1, activation="tanh"):


		# height, width, thickness
		w = "w" + str(conv_layer)
		x_h, x_w, x_t = x.shape[0], x.shape[1], x.shape[2]
		w_h, w_w, w_t, w_f = self.conv_w[w].shape[0], self.conv_w[w].shape[1], self.conv_w[w].shape[2], self.conv_w[w].shape[3]

		h_num = self.calc_num_steps(step, x_h, w_h)
		w_num = self.calc_num_steps(step, x_w, w_w)
		t_num = self.calc_num_steps(step, x_t, w_t)

		#print(h_num, w_num, t_num, w_f)
		#print(w_h, w_w, w_t, w_f)

		self.conv_z["z"+str(conv_layer)] = torch.rand(h_num, w_num, t_num, w_f)

		for h in range(h_num):
			for w in range(w_num):
				for t in range(t_num):
					for f in range(w_f):

						x_slice = x.narrow(0, h*step, w_h).narrow(1, w*step, w_w).narrow(2, t*step, w_t)
						self.conv_z["z"+str(conv_layer)][h, w, t, f] = self.single_conv(x_slice, conv_layer=conv_layer)
						

	def single_pool(self, x, pool_type="average"):

		if pool_type == "average":
			return torch.sum(x)/x.numel()

		if pool_type == "max":
			return x.max()

	def pool_forward(self, x, pool_layer=1, output_shape=[1, 1, 1], step=1, pool_type="average"):
		
		# height, width, thickness
		w = "w" + str(conv_layer)
		x_h, x_w, x_t, x_f = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		h_num, w_num, t_num = output_shape[0], output_shape[1], output_shape[2]

		w_h = self.calc_window_side(x_h, h_num)
		w_w = self.calc_window_side(x_w, w_num)
		w_t = self.calc_window_side(w_t, t_num)


		self.pool_a["a"+str(pool_layer)] = torch.zeros(h_num, w_num, t_num, x_f)

		for h in range(h_num):
			for w in range(w_num):
				for t in range(t_num):
					for f in range(x_f):
						x_slice = x.narrow(0, h*step, w_h).narrow(1, w*step, w_w).narrow(2, t*step, w_t)
						self.conv_a["a"+str(conv_layer)][h, w, t, f] = self.single_pool(x, pool_type)

	def pad(self, a, pad_num = 1):
		padding = [pad_num, pad_num, pad_num, pad_num]
		return torch.nn.functional.pad(a, pad=padding)

	def calc_num_steps(self, step, x, w):
		return int((x-w)/(step))+ 1

	def calc_window_side(step, x, length):
		return  int(x - ( (length - 1) * step ))
	
	def sigmoid_prime(self, z):
		return torch.sigmoid(z) * (1-torch.sigmoid(z))

	def tanh_prime(self, z):
		return 1 - torch.square(torch.tanh(z))

	def dense_forward(self, x):
		pass

	def sigmoid(self, z):
		return torch.sigmoid(z)

	def tanh(self, z):
		return torch.tanh(z)

	def sigmoid_prime(self, z):
		return self.sigmoid(z) * ( 1 - self.sigmoid(z))

	def tanh_prime(self, z):
		return 1 - self.tanh(z)**2

cnn = CNN(2, 4)


x = torch.rand(28, 28, 1)
t = time.time()
cnn.predict(x)
print(time.time() - t)

print(cnn.conv_z["z2"].shape)














