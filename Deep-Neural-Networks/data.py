import pandas as pd
import torch
import numpy as np

def get_data():
    raw_data = pd.read_csv("abalone.csv").to_csv()
    raw_data = raw_data.split("\n")
    data = []
    dataY = []
    for i in range(1, len(raw_data)-1):
      data.append(raw_data[i].split(",")[0:9])
      dataY.append(float(raw_data[i].split(",")[9]))

      """
      if raw_data[i].split(",")[1] == 'I':
        dataY.append([1, 0])

      if raw_data[i].split(",")[1] == 'F':
        dataY.append([0, 1])

      if raw_data[i].split(",")[1] == 'M':
        dataY.append([0, 1])
      """
        

    data = np.array(data)
    Y = np.array(dataY)
    Y = Y/np.max(Y)
    data[data == 'M'] = '1'
    data[data == 'F'] = '1'
    data[data == 'I'] = '1'
    X = torch.from_numpy(data.astype(float)).type(torch.float32)
    Y = torch.from_numpy(Y).type(torch.float32)
    return X, Y
