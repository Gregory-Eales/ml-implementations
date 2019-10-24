import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

from linear_regression.linear_regression import LinearRegressor
from utils.data import generate_data

x, y = generate_data()
x_shuffle, y_shuffle = shuffle(x, y, random_state=0)

lr = LinearRegressor(input_size=2)
lr.train(x_shuffle, y_shuffle, iter=100, alpha=0.0001)

t, s = lr.decision_boundry(start=-0.4, stop=1, step=0.01)

plt.plot(t, s)
plt.scatter(x[0:500,0], x[0:500,1])
plt.scatter(x[500:1000,0], x[500:1000,1])
plt.show()
