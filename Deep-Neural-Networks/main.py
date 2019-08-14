from neural_network import NeuralNetwork
import time
from data import get_data
import torch


x = torch.randn(100, 5)

y = torch.randn(100, 2)


NN = NeuralNetwork(5, 2, 4, gpu=True)

NN.predict(x)
