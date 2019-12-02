import torch
import numpy as np
import gym
import time

from policy_network import PolicyNetwork
from value_network import ValueNetwork
from buffer import Buffer


class PPO(object):

    def __init__(self, alpha, in_dim, out_dim):

        self.out_dim = out_dim
        self.in_dim = in_dim

        self.buffer = Buffer()
        self.value_net = ValueNetwork(alpha, in_dim, 1)
        self.policy_net = PolicyNetwork(alpha, in_dim, out_dim)
        self.old_policy_net = PolicyNetwork(alpha, in_dim, out_dim)


    def store(self, state, action, reward):

        self.buffer.store_state(state)
        self.buffer.store_action(action)
        self.buffer.store_reward(reward)

    def calculate_advantages(self):

        s2 = torch.Tensor(self.buffer.states[-1])
        s1 = torch.Tensor(self.buffer.states[-2])

        v = self.value_net(s1)
        q = self.value_net(s2)
        a = q-v+1

        self.buffer.store_advantage(a)

    def update(self):
        state, action, reward, disc_reward, advantage = self.buffer.get_data()

    def get_action(self):
        state = torch.Tensor(self.buffer.states[-1]).reshape(1, 3)
        prediction = self.policy_net.forward(state)
        old_pred = self.old_policy_net.forward(state)
        self.buffer.store_action(prediction)
        self.buffer.store
        action_probabilities = torch.distributions.Categorical(prediction)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        self.buffer.log_probs.append(log_prob)
        # convert discrete to continuous2
        action = -2.0 + 0.1*action.float()



        return action.item()

    def train(self, env, n_steps, n_epoch, render=False, verbos=False):

        for i in range(n_epoch):

            # reset env
            state = env.reset()
            self.buffer.store_state(state)

            for s in range(n_steps):

                if render: self.env.render()

                # render if true
                if render: env.render()

                # get action
                action = self.get_action()

                # get observation
                state, reward, done, info = env.step([0])

                # store metrics
                if not done: self.store(state, action, reward)

                if done or s == n_steps-1:
                    self.buffer.store_action(action)
                    self.buffer.store_reward(reward)
                    self.buffer.discount_rewards(s+2)

            # update networks
            #self.update()



            print(len(self.buffer.states))
            print(len(self.buffer.actions))
            print(len(self.buffer.rewards))
            print(len(self.buffer.log_probs))
            print(len(self.buffer.discounted_rewards))
            print(len(self.buffer.advantages))


def main():
    env = gym.make("Pendulum-v0")

    ppo = PPO(alpha=0.01, in_dim=3, out_dim=21)

    ppo.train(env=env, n_steps=10, n_epoch=3)


if __name__ == "__main__":
    main()
