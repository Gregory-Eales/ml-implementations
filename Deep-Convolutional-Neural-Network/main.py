from cnn import CNN
from data import get_MNIST_data
import torch
import time
from tqdm import tqdm


print(torch.__version__)


# Get Data
"""

x, y = get_MNIST_data()

x_sample = torch.tensor(x[0].reshape(8, 8, 1))
y_sample = torch.tensor(y[0])

"""


# Load Model



cnn = CNN(2, 2)

x = torch.rand(8, 8, 1)
y = torch.rand(1, 10)


torch.matmul(x, y)

t = time.time()

cnn.train(x, y, iterations=1, alpha=0.01)

print(time.time() - t)
