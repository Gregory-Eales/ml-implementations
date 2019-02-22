import data
import logistic_regression

regressor = logistic_regression.LogisticRegressor([1, 2])
regressor.train(1, 2, 100, 1)