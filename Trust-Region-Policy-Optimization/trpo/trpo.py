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

    def calculate_advantage(self, prev_observation, observation):

        v1 = self.value_network(prev_observation)

        v2 = self.value_network(observation)

        return 1 + v2 - v1

    def act(self, observation):
        pass

    def train(self, env, epochs=1000, steps=4000):

        for epoch in range(epochs):

            observation = env.reset()
            step = 0

            for step in range(steps):

                step += 1

                action, log_prob = self.act()

                observation, reward, done, info = env.step(action)

                if done:
                    observation = env.reset()
                    step = 0
