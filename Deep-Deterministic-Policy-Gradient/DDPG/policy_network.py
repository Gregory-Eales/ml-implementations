import torch
from torch.nn import functional as F
from torch import optim
from torch import nn

class PolicyNetwork(torch.nn.Module):

	def __init__(self, in_dim, out_dim, alpha=0.01):

		super(PolicyNetwork, self).__init__()

		self.in_dim = in_dim
		self.out_dim = out_dim

		self.l1 = nn.Linear(in_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, out_dim)

		self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
		self.to(self.device)

	def forward(self, x):
		
		out = torch.Tensor(x).reshape(-1, self.in_dim)
		out = self.l1(out)
		out = F.relu(out)
		out = self.l2(out)
		out = F.relu(out)
		out = self.l3(out)
		out = torch.tanh(out)

		return out


	def loss(self, q):
		return torch.mean(q)

	def optimize(self, q):

	  torch.cuda.empty_cache()
	  self.optimizer.zero_grad()
	  loss = self.loss(q)
	  loss.backward(retain_graph=True)
	  self.optimizer.step()

def main():
	pn = PolicyNetwork(in_dim=3, out_dim=1)
	x = torch.ones(10, 3)
	print(pn.forward(x))

if __name__ == "__main__":
	main()
