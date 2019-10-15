import torch
import numpy

from policy_network import PolicyNetwork
from value_network import ValueNetwork


class TRPO(object):

    def __init__(self, alpha, input_size, output_size):

        self.value_network = ValueNetwork(alpha, input_size=input_size,
         output_size=1)

        self.policy_network = PolicyNetwork(alpha, input_size=input_size,
         output_size=output_size)

    def train(self, env, epochs=1000, steps=4000):


        for epoch in range(epochs):

            for step in range(steps):
                pass
