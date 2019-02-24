import numpy as np
import random


def gen_gaussian_data(num_points = 50):
	x = []
	y = []

	for i in range(int(num_points/2)):
		x.append([random.gauss(10, 2), random.gauss(5, 2)])
		y.append([1])

	for i in range(int(num_points/2)):
		x.append([random.gauss(2, 1), random.gauss(10, 2)])
		y.append([0])
		

	x = np.array(x)
	y = np.array(y)

	return x, y