import torch
import numpy as np

from policy_network import PolicyNetwork
from value_network import ValueNetwork
from buffer import Buffer


class PPO(object):

    def __init__(self, alpha, input_size, output_size):

        self.buffer = Buffer()

        self.value_network = ValueNetwork(alpha=alpha, input_size=input_size+3)

        self.policy_network = PolicyNetwork(alpha=alpha, input_size=input_size,
         output_size=output_size)

        self.old_policy_network = PolicyNetwork(alpha=alpha,
         input_size=input_size, output_size=output_size)

        # store policy state
        self.buffer.store_parameters(self.policy_network.state_dict())

    def update(self, iter=80):

        observations = self.buffer.get_observations()
        rewards = self.buffer.get_rewards()
        advantages = self.buffer.get_advantages()
        log_probs = self.buffer.get_log_probs()

        self.old_policy_network.load_state_dict(self.buffer.params)
        old_probs = self.old_policy_network.forward(observations)
        action_probabilities = torch.distributions.Categorical(old_probs)
        old_actions = action_probabilities.sample()
        old_probs = action_probabilities.log_prob(old_actions)
        self.buffer.store_parameters(self.policy_network.state_dict())

        self.policy_network.optimize(log_probs, old_probs,
         advantages, 0.01)

        self.value_network.optimize(actions, observations, rewards, iter=iter)

    def calculate_advantage(self):

        
        prev_observation = self.buffer.observation_buffer[-2]
        observation = self.buffer.observation_buffer[-1]
        prev_log_prob = self.buffer.log_probs[-2]
        log_prob = self.buffer.log_probs[-1]
        prev_input = torch.cat([prev_log_prob, prev_observation])
        input = torch.cat([log_prob, observation])

        v1 = self.value_network(prev_observation)
        v2 = self.value_network(observation)

        return 1 + v2 - v1


    def act(self, s):
        s = torch.tensor(s).reshape(-1, len(s)).float()
        prediction = self.policy_network.forward(s)
        action_probabilities = torch.distributions.Categorical(prediction)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        return action.item(), log_prob

    def discount_rewards(self, step):
        for s in reversed(range(1, step+1)):
            update = 0
            for k in reversed(range(1, s+1)):
                update += self.buffer.reward_buffer[-k]*(0.99**k)
            self.buffer.reward_buffer[-s] += update

    def train(self, env, epochs=1000, steps=4000):

        rewards = []

        for epoch in range(epochs):

            observation = env.reset()
            self.buffer.store_observation(observation)
            step = 0

            for step in range(steps):

                step += 1

                action, log_prob = self.act(observation)
                self.buffer.store_log_prob(log_prob)

                observation, reward, done, info = env.step(action)
                self.buffer.store_observation(observation)
                self.buffer.store_reward(reward)

                advantage = self.calculate_advantage()
                self.buffer.store_advantage(advantage)

                if done or step == steps-1:
                    observation = env.reset()
                    self.discount_rewards(step)
                    step = 0
                    self.buffer.reward_buffer[0]

            self.update()

def main():
    import gym
    torch.manual_seed(1)
    np.random.seed(1)
    env = gym.make('MountainCar-v0')

    trpo = PPO(alpha=0.001, input_size=2, output_size=3)
    trpo.train(env=env, epochs=1, steps=1000)


if __name__ == "__main__":
    main()
