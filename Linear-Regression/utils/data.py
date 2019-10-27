import numpy as np
from matplotlib import pyplot as plt

def generate_data():

    a = np.random.normal(loc=25, scale=6, size=500)
    b = np.random.normal(loc=5, scale=8, size=500)

    x1 = np.array([a, b]).T
    x2 = np.array([b, a]).T
    y1 = np.ones([500, 1])
    y2 = np.zeros([500, 1])

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    # normalize
    x = x/np.max(x)

    return x, y
