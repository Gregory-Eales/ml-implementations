import data
import logistic_regression
import numpy as np
from matplotlib import pyplot as plt
import random


x = []
y = []

for i in range(20):
	num1 = random.gauss(10, 0.3)
	num2 = random.gauss(1, 0.3)
	x.append([num2, num1])
	y.append([1])

for i in range(20):
	num1 = random.gauss(5, 0.3)
	num2 = random.gauss(5, 0.3)
	x.append([num2, num1])
	y.append([0])

x = np.array(x)
y = np.array(y)




regressor = logistic_regression.LogisticRegressor([1, 2])

regressor.train(x, y, 50000, 0.01)



print(regressor.predict(np.array([5, 5])))
print(regressor.predict(np.array([3, 10])))



db1 = []
db2 = []


for i in range(200):
	for j in range(200):
		prediction = regressor.predict(np.array([i/20, j/20]))
		if prediction < 0.51 and prediction > 0.49:
			db1.append(i/20)
			db2.append(j/20)



#plt.plot(regressor.historical_cost)
plt.scatter(db1, db2)
plt.scatter(x[:, 0], x[:, 1])
plt.show()






