from perceptron.perceptron import Perceptron
from utils.abalone_data import get_abalone
from matplotlib import pyplot as plt


x, y = get_abalone()

model = Perceptron(input_shape = [1, 9])

model.train(x, y, 0.005, 100)

plt.plot(model.historical_error)
plt.show()
