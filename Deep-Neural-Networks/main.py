from neural_network import NeuralNetwork 
import time



nn = NeuralNetwork()

t = time.time()

nn.tf_sigmoid(2.000)

print(time.time() - t)


t = time.time()

nn.np_sigmoid(2.000)

print(time.time() - t)


t = time.time()

nn.reg_sigmoid(2.000)

print(time.time() - t)
