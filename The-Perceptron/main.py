from perceptron import Perceptron
from abalone_data import get_abalone


x, y = get_abalone()

model = Perceptron(input_shape = [1, 9])

model.train(x, y, 0.0001, 1)

