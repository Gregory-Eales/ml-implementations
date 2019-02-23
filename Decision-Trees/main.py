from matplotlib import pyplot as plt 
from decision_tree import DecisionTree
import numpy as np
from gaussian_data import gen_gaussian_data
import random

x, y = gen_gaussian_data()


model = DecisionTree()

model.train(x, y, alpha, iterations)

model.predict(x)


plt.scatter(x[:, 0], x[:, 1])
plt.show()