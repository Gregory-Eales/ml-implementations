import torch


class ValueNetwork(torch.nn.Module):

    def __init__(self, alpha, in_dim, out_dim):

        super(ValueNetwork, self).__init__()
