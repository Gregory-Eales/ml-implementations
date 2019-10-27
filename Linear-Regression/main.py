import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

from linear_regression.linear_regression import LinearRegressor
from utils.data import generate_data

x, y = generate_data()
x_shuffle, y_shuffle = shuffle(x, y, random_state=0)

lr = LinearRegressor(input_size=2)
lr.train(x_shuffle, y_shuffle, iter=10000, alpha=0.001)
plt.title("Loss per Iteration")
plt.xlabel("Iteration(s)")
plt.ylabel("Loss")
plt.plot(lr.historical_error)
plt.show()

t, s = lr.decision_boundry(start=-0.4, stop=1, step=0.01)

plt.title("Accuracy per Epoch")
plt.plot(lr.historical_accuracy)
plt.xlabel("Iteration(s)")
plt.ylabel("Accuracy")
plt.show()

plt.title('Linear Classification')
plt.plot(t, s, label='Decision Boundry')
plt.scatter(x[0:500,0], x[0:500,1], label='Class 1', marker='x')
plt.scatter(x[500:1000,0], x[500:1000,1], label='Class 2', marker='v')
plt.legend(['Decision Boundry', 'Class 1', 'Class 2'], loc='lower left')
plt.show()
