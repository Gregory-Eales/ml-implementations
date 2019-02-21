import numpy as np
import time

class Perceptron(object):
	
	def __init__(self, input_shape = [1, 2]):
		self.w = np.random.random(x.shape)

	def train(self, x, y, alpha, iterations):
		
		for i in range(x.shape[0]):
			z = np.dot(self.w.T, x[i])
			
			if z < 0.5:
				z = 0

			else:
				z = 1

			self.w = self.w - self.cost(y, z, x[i], alpha)

	def cost(self, z, y, x, alpha):
		return alpha * x * (z - y)
		

	def predict(self, x):
		z = np.dot(self.w.T, x[i])
		if z < 0.5:
			return 0

		else:
			return 1

		




