import numpy as np
from matplotlib import pyplot as plt

# decision tree classifier
class DecisionTree(object):

	def __init__(self, depth=2):
		pass

	def train(self, x):
		pass

	def get_purity(self, x):

		types = []

		count = {}

		for i in range(len(x)):

			if x[i][-1] not in types:
				types.append(x[i][-1])

		for t in types:
			count[str(t)] = 0

		for i in range(len(x)):
			for t in types:
				if x[i][-1] == t:
					count[str(t)] = count[str(t)] + 1

		max_num = [1, "string"]

		for i in count:
			if max_num[0] < count[i]:
				max_num = [count[i], i]

		return round(max_num[0]/len(x), 4), max_num[1]


