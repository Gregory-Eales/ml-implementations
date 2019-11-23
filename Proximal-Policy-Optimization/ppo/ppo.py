import torch
import numpy as np

from policy_network import PolicyNetwork
from value_network import ValueNetwork
from buffer import Buffer


class PPO(object):

    def __init__(self, alpha, in_dim, output_dim):


        self.buffer = Buffer()
        self.value_net = ValueNetwork()
        self.policy_net = PolicyNetwork()
