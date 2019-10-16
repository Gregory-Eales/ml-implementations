import torch
import numpy as np

from policy_network import PolicyNetwork
from value_network import ValueNetwork
from buffer import Buffer


class TRPO(object):

    def __init__(self, alpha, input_size, output_size):

        self.buffer = Buffer()

        self.value_network = ValueNetwork(alpha, input_size=input_size,
         output_size=1)

        self.policy_network = PolicyNetwork(alpha, input_size=input_size,
         output_size=output_size)

    def calculate_advantage(self, prev_observation, observation):

        v1 = self.value_network(prev_observation)

        v2 = self.value_network(observation)

        return 1 + v2 - v1

    def act(self, observation):
        prediction = self.policy_network.forward(s)
        action_probabilities = torch.distributions.Categorical(prediction)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        return action.item(), log_prob

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

def main():

    import gym
    torch.manual_seed(1)
    np.random.seed(1)
    env = gym.make('MountainCar-v0')
    observation = env.reset()
    print(type(observation.tolist()))

if __name__ == "__main__":
    main()
