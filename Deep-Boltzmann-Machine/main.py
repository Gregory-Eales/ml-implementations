import torch
import numpy as np
from tqdm import tqdm 


class DBM(torch.nn.Module):

	def __init__(self):
		

		self.optimizer = torch.optim.Adam()

	def forward(self, x):
		pass

	def loss(self, x):
		pass

	def optimize(self, x, y, epochs):
		
		
		for i in tqdm(range(epochs), "Training Model"):
			pass
			
	



