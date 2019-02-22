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
		for iteration in tqdm(range(iterations)):
			sleep(0.1)

	def cost(self):
		pass

	def cost_prime(self):
		pass

	def sigmoid(self):
		pass

	def sig_prime(self):
		pass

	def predict(self):
		pass