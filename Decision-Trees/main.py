from matplotlib import pyplot as plt 
from decision_tree import DecisionTree
import numpy as np
from gaussian_data import gen_gaussian_data
import random

x, y = gen_gaussian_data(num_points = 100)


model = DecisionTree()

model.train(x, y, alpha=0.1, iterations=100)

model.predict(x)


plt.scatter(x[0:50:, 0], x[0:50, 1])
plt.scatter(x[50:, 0], x[50:, 1])
plt.show()