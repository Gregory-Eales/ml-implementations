import numpy as np
import pandas as pd
import io
import requests


def get_abalone():

	to_predict = "Height"

	url="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
	s=requests.get(url).content
	abalone_dataset = pd.read_csv(io.StringIO(s.decode('utf-8')))
	abalone_dataset.columns = ['Gender', "Length", "Diameter", "Height", 
		"Whole Weight", "Shucked Weight", "Viscera Weight", "Shell Weight",
		 "Rings"]

	abalone_dataset["Gender"] = np.where(abalone_dataset["Gender"] != "M", 0,
	 abalone_dataset["Gender"])

	abalone_dataset['Gender'] = np.where(abalone_dataset['Gender'] == "M", 1,
	 abalone_dataset['Gender'])

	y = np.array(abalone_dataset[to_predict].values)
	abalone_dataset[to_predict] = np.where(abalone_dataset[to_predict] == 0,
	 0, abalone_dataset[to_predict])
	x = np.array(abalone_dataset.values)

	return x, y