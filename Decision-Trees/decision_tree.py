import numpy as np
from matplotlib import pyplot as plt

# decision tree classifier
class DecisionTree(object):

	# need to implement tree depth, pruning, purity of classification
	# recursivly split data to find which is the best
	# other methods besides brute force?
	
	def __init__(self, indxs=1, min_leaf=1, depth=1):
		self.indxs = indxs
		self.min_leaf = min_leaf
		self.depth = depth

	def create_tree(self):
		conditons = []
		for i in range(self.depth):
			conditions.append()

	def train(self, x, y, alpha, iterations):
		
		for i in range(iterations):
			for j in range(x.shape[0]):
				pass

	# makes classification prediction
	def predict(self, x):
		pass

	# randomly split data into sections
	# create branch of the tree
	def get_split(data):

		group_a = []
		group_b = []

		condtion = True

		for indx in range(self.indxs):
			if condition:
				group_a.append(data[indx])

			else:
				group_b.append(data[indx])

	# puts prediction to the terminal
	def to_terminal(self):
		pass

	# finds the maximum purity value
	def purity(self, predictions, y):
		classes = []
		for i in range(predictions):
			if y[i] not in classes:
				classes.append(y[i])

		for clss in classes:
			counter = 0
			for i in range(predictions):
				if clss == y[i]:
					counter = counter + 1

