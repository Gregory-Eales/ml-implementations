from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from perceptron.perceptron import Perceptron
from utils.abalone_data import get_abalone


x, y = get_abalone()

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

model = Perceptron(input_shape = [1, 9])

model.train(x_train, y_train, 0.0005, 100)


plt.plot(model.historical_error)
plt.show()
