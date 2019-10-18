import time

import torch
from matplotlib import pyplot as plt
import numpy as np

from dnn.neural_network import NeuralNetwork
from utils.data import get_data

x, y = get_data()


NN = NeuralNetwork(5, 3, 5, hidden_addition=1)


NN.train(x, y, iterations=50000, alpha=0.001)

plt.plot(NN.historical_cost)

plt.show()



predictions = NN.predict(x).cpu().detach().numpy()
answers = y.numpy()

p = np.zeros_like(predictions)
p[np.arange(len(predictions)), predictions.argmax(1)] = 1

print(answers[0:5])
print(p[0:5])
p = p.tolist()
answers = answers.tolist()
correct = 0
for i in range(len(p)):
  if p[i] == answers[i]:
    correct += 1

print(correct)

print("The model is " + str((correct/len(p))*100) + "% accurate")
