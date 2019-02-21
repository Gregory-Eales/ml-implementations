from perceptron import Perceptron
from abalone_data import get_abalone
from matplotlib import pyplot as plt


x, y = get_abalone()

model = Perceptron(input_shape = [1, 9])

model.train(x, y, 0.005, 10)

for i in range(100):
	print(model.predict(x[i]), y[i])

plt.plot(model.historical_error)
plt.show()
