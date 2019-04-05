from cnn import CNN
from data import get_MNIST_data
import torch
import time
from tqdm import tqdm


# Get Data
"""
x, y = get_MNIST_data()

x_sample = x[0]

"""

# Load Model


cnn = CNN(2, 4)

x = torch.rand(8, 8, 1)
t = time.time()

for i in tqdm(range(500)):
    cnn.predict(x)

print(time.time() - t)

print(cnn.conv_z["z2"].shape)