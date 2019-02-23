from matplotlib import pyplot as plt 
from decision_tree import DecisionTree
import numpy as np
import random


x = []
y = []

for i in range(50):
	x.append([random.gauss(10, 2), random.gauss(5, 2)])
	y.append([1])
	x.append([random.gauss(2, 1), random.gauss(10, 3)])
	y.append([0])

x = np.array(x)
y = np.array(y)

model = DecisionTree()

model.train(x, y, alpha, iterations)

model.predict(x)


plt.scatter(x[:, 0], x[:, 1])
plt.show()