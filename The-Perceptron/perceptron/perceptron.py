import numpy as np
import time
from tqdm import tqdm

class Perceptron(object):

	def __init__(self, input_shape = [1, 2]):
		self.w = np.random.random(input_shape) / 10
		self.w = self.w[0]
		self.error = 0
		self.historical_error = []

	def train(self, x, y, alpha, iterations):

		for iteration in tqdm(range(iterations)):
			self.error = 0

			for i in range(x.shape[0]):
				z = np.dot(self.w, x[i].T)
				self.error = self.error + abs(y[i] - z)
				self.w = self.w - self.cost(y[i], z, x[i], alpha)

			self.historical_error.append(self.error)

	def cost(self, z, y, x, alpha):
		return alpha * x * (y - z)

	def predict(self, x):
		z = np.dot(self.w, x.T)
		return z
