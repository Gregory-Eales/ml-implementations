import torch

from utils.generator import generate_bit_strings
from network.network import Network


# generate data
bit_strings, parities = generate_bit_strings(1000, 50)


x = torch.tensor(bit_strings)
y = torch.tensor(parities).reshape(-1, 1)

print(x.shape)
print(y.shape)

net = Network()
