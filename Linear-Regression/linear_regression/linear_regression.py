import numpy as np
from matplotlib import pyplot as plt

class LinearRegressor(object):

    def __init__(self, input_size):
        self.w = np.zeros([input_size, 1])
        print(self.w.shape)
        self.b = np.zeros([1, 1])
        self.historical_error = []
        self.historical_accuracy = []

    def mean_squared_error(self, prediction, y):
        return 0.5*np.sum((prediction - y)**2) / y.shape[0]

    def mse_prime(self, prediction, y):
        return prediction-y

    def loss(self, x, prediction, y):
        loss = self.mse_prime(prediction, y)
        delta_w = np.sum(x*loss, axis=0).reshape(2, 1)/x.shape[0]
        delta_b = np.sum(loss)/x.shape[0]
        return delta_w, delta_b

    def predict(self, x):
        return np.dot(x, self.w) + self.b

    def train(self, x, y, iter, alpha):

        for i in range(iter):
            prediction = self.predict(x)
            error = self.mean_squared_error(prediction, y)
            self.historical_error.append(error)
            delta_w, delta_b = self.loss(x, prediction, y)
            self.w = self.w - alpha*delta_w
            self.b = self.b - alpha*delta_b
            self.historical_accuracy.append(self.get_accuracy(x, y))

    def decision_boundry(self, start, stop, step):

        b = (self.b[0] - 0.5)/(-self.w[0])
        m = (self.w[1])/(-self.w[0])
        t = np.arange(start, stop, step)
        s = m*t + b
        return t, s

    def get_accuracy(self, x, y):
        predictions = self.predict(x)
        predictions[predictions<0.5] = 0
        predictions[predictions>=0.5] = 1
        total_samples = y.shape[0]
        actual_pred = y
        pred_error = np.sum(np.absolute(predictions-y))
        return 100*(total_samples-pred_error)/total_samples


def main():
    x, y = generate_data()
    x_shuffle, y_shuffle = shuffle(x, y, random_state=0)
    lr = LinearRegressor(2)
    print(lr.w)
    lr.train(x, y, iter=5000, alpha=0.0001)
    plt.plot(lr.historical_error)
    plt.show()
    lr.get_accuracy(x, y)


if __name__ == "__main__":
    main()
