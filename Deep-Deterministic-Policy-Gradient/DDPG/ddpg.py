import torch
from buffer import Buffer
from policy_network import PolicyNetwork
from q_network import QNetwork


class DDPG(object):

    def __init__(self):

        self.buffer
