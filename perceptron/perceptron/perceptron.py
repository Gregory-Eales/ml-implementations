import numpy as np
import time
from tqdm import tqdm

class Perceptron(object):

	def __init__(self, input_shape = [1, 2]):
		self.w = np.random.random(input_shape) / 10
		self.w = self.w[0].reshape(1, -1)
		self.error = 0
		self.historical_error = []

	def train(self, x, y, alpha, iterations):

		m = x.shape[0]

		for iteration in tqdm(range(iterations)):

			z = self.predict(x)
			self.error = abs(y - z)
			self.w = self.w - np.sum(self.cost(y, z, x, alpha).T, axis=1)/m
			self.historical_error.append(np.sum(self.error))

	def cost(self, z, y, x, alpha):
		cost = alpha * x * (y - z)
		return cost

	def predict(self, x):
		z = np.dot(self.w, x.T)
		z[z>=0] = 1
		z[z<0] = 0
		return z.reshape((-1, 1))
