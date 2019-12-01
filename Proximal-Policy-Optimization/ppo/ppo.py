import torch
import numpy as np

from policy_network import PolicyNetwork
from value_network import ValueNetwork
from buffer import Buffer


class PPO(object):

    def __init__(self, alpha, in_dim, out_dim):

        self.buffer = Buffer()
        self.value_net = ValueNetwork(alpha, in_dim, out_dim)
        self.policy_net = PolicyNetwork(alpha, in_dim, out_dim)


    def store(self, state, action, reward):

        self.buffer.store_state(state)
        self.buffer.store_action(action)
        self.buffer.store_reward(reward)

    def calc_discounted_rewards(self, n_steps):
        # calculates the discounted reward using the past n rewards, where
        # n is equal to the number of steps
        self.buffer.discount_reward(n_steps)

    def train(self, env, n_steps, n_epoch, render=False, verbos=False):


        for i in range(iter):

            # reset env
            state = env.reset()

            for s in range(nume_steps):

                # render if true
                if render: env.render()

                # get action
                action = self.get_action()

                # get observation
                state, reward, done, info = env.step(action)

                # store metrics
                self.store(state, action, reward)

            # update networks
