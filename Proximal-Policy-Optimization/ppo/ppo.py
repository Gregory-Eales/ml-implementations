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

    def train(self, env, num_steps, iter):


        for i in range(iter):

            # reset env
            env.reset()

            for s in range(nume_steps):

                # get observation

                # take action

                # get reward
                pass

            # update networks
