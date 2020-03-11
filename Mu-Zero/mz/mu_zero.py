import torch
from policy import PolicyNet
from value import ValueNet


class MuZero(torch.nn.Module):

    def __init__(self):

        self.policy_net = PolicyNet()
        self.value_net = ValueNet()


    def update(self):
        pass


    def act(self):
        pass
