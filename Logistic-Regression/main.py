import numpy as np
from matplotlib import pyplot as plt
from random import gauss


def get_data(mean, sigma, classification):
    x = []
    y = []
    for i in range(60):
        dat_x, dat_y = gauss(mean, sigma), gauss(mean, sigma)
        x.append([1, dat_x, dat_y])
        y.append([classification])
    x = np.array(x)
    y = np.array(y)

    return x, y


def array_combiner(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i])
    for i in range(len(b)):
        c.append(b[i])

    return np.array(c)


def sigmoid(z):
    sig = 1/(1 + np.exp(-z))
    return sig


def j_cost(theta, x, y):
    z = np.dot(x, theta)
    cost = np.sum(y*z - np.log(1 + np.exp(z)))
    return cost


def train(theta, x, y, alpha):
    j_hist = []
    for i in range(10000):
        cost = j_cost(theta, x, y)
        cost = abs(cost)
        j_hist.append(cost)
        z = np.dot(x, theta)
        predictions = sigmoid(z)
        error = y-predictions
        gradient = np.dot(x.transpose(), error)
        theta += alpha * gradient
        if abs(cost) < 0.05:
            break
    print(cost)
    return j_hist, theta


x1, y1 = get_data(4, 0.4, 1)
x2, y2 = get_data(2, 0.4, 0)
plt.scatter(x1[:,1], x1[:,2], marker="x", label='1' )
plt.scatter(x2[:,1], x2[:,2], marker="o", label='0')
plt.title("Logistic Regression")
plt.legend(loc="upper left")
plt.ylabel("Y Variable")
plt.xlabel("X Variable")
plt.show()
x = (array_combiner(x1, x2))
y = (np.append(y1, y2))
theta = np.zeros(x.shape[1])
alpha = 0.001
j, theta = train(theta, x, y, alpha)
plt.legend(loc='upper left')
plt.plot(j)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Training")
plt.show()


xDecision = []
yDecision = []

for i in range(150):
    for j in range(150):
        z = np.dot(np.array([1, i/20, j/20]), theta)
        prediction = sigmoid(z)
        if prediction < 0.51 and prediction > 0.49:
            xDecision.append(i/20)
            yDecision.append(j/20)

plt.plot(xDecision, yDecision, label="Decision Boundary")
plt.scatter(x1[:,1], x1[:,2], marker="x", label='1')
plt.scatter(x2[:,1], x2[:,2], marker="o", label='0')
plt.title("Logistic Regression")
plt.legend(loc="upper left")
plt.show()