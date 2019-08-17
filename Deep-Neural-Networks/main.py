from neural_network import NeuralNetwork
import time
from data import get_data
import torch
from matplotlib import pyplot as plt
import numpy as np


x, y = get_data()

print(x[1:3])

print(y[1:3])


NN = NeuralNetwork(9, 1, 4, hidden_addition=3)


NN.train(x[0:3000], y[0:3000], iterations=5000, alpha=0.01)

plt.plot(NN.historical_cost)

plt.show()



predictions = NN.predict(x[3000:4000]).cpu().detach().numpy()
answers = y[3000:4000].numpy()

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