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
        prediction = self.policy_network.forward(observation)
        action_probabilities = torch.distributions.Categorical(prediction)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        return action.item(), log_prob

    def train(self, env, epochs=1000, steps=4000):

        for epoch in range(epochs):

            observation = env.reset()
            self.buffer.store_observation(observation)

            step = 0

            for step in range(steps):

                step += 1

                action, log_prob = self.act(observation)
                self.buffer.store_action(log_prob)

                observation, reward, done, info = env.step(action)
                self.buffer.store_observation(observation)
                self.buffer.store_reward(reward)

                advantage = self.calculate_advantage()
                self.buffer.store_advantage(advantage)

                if done or t == n_steps-1:
                    observation = env.reset()

                    for s in reversed(range(1, step+1)):
                        update = 0
                        for k in reversed(range(1, s+1)):
                            update += self.buffer.reward_buffer[-k]*(0.99**k)
                        self.buffer.reward_buffer[-s] += update
                    step = 0
def main():
    import gym
    torch.manual_seed(1)
    np.random.seed(1)
    env = gym.make('MountainCar-v0')

    trpo = TRPO(alpha=0.001, input_size=2, output_size=3)
    trpo.train(env, epochs=1, steps=200)

if __name__ == "__main__":
    main()
