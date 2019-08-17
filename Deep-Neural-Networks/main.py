from neural_network import NeuralNetwork
import time
from data import get_data
import torch
from matplotlib import pyplot as plt


x, y = get_data()

print(x[1:3])

print(y[1:3])

NN = NeuralNetwork(8, 3, 5)

NN.train(x, y, iterations=100000, alpha=0.05)

plt.plot(NN.historical_cost)

#plt.show()


print(NN.predict(x[1:3]))
