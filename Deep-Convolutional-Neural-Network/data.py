# import dependencies
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt


# load digit data set
x, uncleaned_y = datasets.load_digits(return_X_y=True)

x = x/10
# create a y for each classification: numbers 0-9 and stores it in 'answers'
answers = []
for i in range(10):
  zero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  zero[i] = 1
  answers.append(zero)

# iterate through 'uncleaned_y' and add the correct classification for each y
y = []
for i in uncleaned_y:
  for j in range(10):
    if i == j:
      y.append(answers[j])

# convert y to an array      
y = np.array(y)

def get_data():
  pass


