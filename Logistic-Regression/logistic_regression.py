import numpy as np 
from tqdm import tqdm
from time import sleep

class LogisticRegressor(object):

	def __init__(self, input_dim=[1, 2]):
		self.w = np.random.random(input_dim)

	def train(self, x, y, iterations, alpha):
		print("    ------------------------------------------")
		print("    |   Training Logistic Regression Model   |")
		print("    ------------------------------------------")
		for i in tqdm(range(iterations)):
			z = np.dot(self.w, x.T)
			a = self.sigmoid(z)

	def cost(self, y_hat, y):
		return y*np.log(y_hat) - (1-y)*np.log(1-y_hat)

	def cost_prime(self):
		pass
  
	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def sig_prime(self, z):
		return self.sigmoid(z)*(1 - self.sigmoid(z))

	def predict(self, x):
		return self.sigmoid(np.dot(self.w, x.T))
		