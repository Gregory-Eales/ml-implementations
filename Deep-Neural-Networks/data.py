import pandas as pd
import torch
import numpy as np

def get_data():
    raw_data = pd.read_csv("iris.csv").to_csv()
    raw_data = raw_data.split("\n")
    data = []
    dataY = []
    for i in range(1, len(raw_data)-1):
     
      data.append(raw_data[i].split(",")[0:5])



      
      if raw_data[i].split(",")[5] == 'Iris-virginica':
        dataY.append([1, 0, 0])

      elif raw_data[i].split(",")[5] == 'Iris-setosa':
        dataY.append([0, 1, 0])

      elif raw_data[i].split(",")[5] == 'Iris-versicolor':
        dataY.append([0, 0, 1])
      
        

    data = np.array(data)
    Y = np.array(dataY)
    data[data == 'Iris-virginica'] = '1'
    data[data == 'Iris-versicolor'] = '1'
    data[data == 'Iris-setosa'] = '1'
    X = torch.from_numpy(data.astype(float)).type(torch.float32)
    Y = torch.from_numpy(Y).type(torch.float32).view(Y.shape[0], -1)
    return X, Y
