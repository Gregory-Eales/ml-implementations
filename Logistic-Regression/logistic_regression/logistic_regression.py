import numpy as np
from tqdm import tqdm
from time import sleep

class LogisticRegressor(object):

	def __init__(self, input_dim=[1, 2]):
		self.w = np.random.random(input_dim)[0]
		self.b = 0
		self.historical_cost = []

	def train(self, x, y, iterations, alpha):
		
		for iteration in tqdm(range(iterations)):
			error = 0
			for i in range(x.shape[0]):
				z = np.dot(self.w, x[i].T) + self.b
				a = self.sigmoid(z)
				error = error + self.cost(a, y[i])
				self.w = self.w - alpha * x[i] * self.sig_prime(z) * self.cost(a, y[i])
				self.b = self.b -  alpha * self.sig_prime(z) * self.cost(a, y[i])
			self.historical_cost.append(error)

	def cost(self, y_hat, y):
		return y_hat - y

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def sig_prime(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def predict(self, x):
		return self.sigmoid(np.dot(self.w, x.T))
