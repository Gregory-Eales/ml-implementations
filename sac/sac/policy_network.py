import torch
from torch.nn import functional as F
from torch import optim
from torch import nn
from torch.distributions.normal import Normal
import numpy as np

class PolicyNetwork(torch.nn.Module):

	def __init__(self, in_dim, out_dim, alpha=0.01):

		super(PolicyNetwork, self).__init__()

		self.in_dim = in_dim
		self.out_dim = out_dim

		self.l1 = nn.Linear(in_dim, 128)
		self.l2 = nn.Linear(128, 128)
		self.l3 = nn.Linear(128, 64)
		self.l4 = nn.Linear(64, out_dim)

		self.log_linear = nn.Linear(out_dim, out_dim)
		self.mu_linear = nn.Linear(out_dim, out_dim)

		self.leaky_relu = nn.LeakyReLU()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

		self.sigmoid = nn.Sigmoid()

		self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
		self.to(self.device)

	def forward(self, x):
		
		out = torch.Tensor(x).reshape(-1, self.in_dim)

		out = self.l1(out)
		out = self.leaky_relu(out)
		out = self.l2(out)
		out = self.leaky_relu(out)
		out = self.l3(out)
		out = self.leaky_relu(out)
		out = self.l4(out)
		out = self.tanh(out)

		return out

	def log_forward(self, x):

		out = torch.Tensor(x).reshape(-1, self.in_dim)

		out = self.l1(out)
		out = self.leaky_relu(out)
		out = self.l2(out)
		out = self.leaky_relu(out)
		out = self.l3(out)
		out = self.leaky_relu(out)
		out = self.l4(out)
		#out = self.tanh(out)
		
		
		mu = self.mu_linear(out)
		log_std = self.log_linear(out)

		log_std = torch.clamp(log_std, -20, 2)
		std = torch.exp(log_std)
		distribution = Normal(mu, std)

		action = distribution.rsample()
		log_p = distribution.log_prob(action)
		log_p -= (2*(np.log(2) - action - F.softplus(-2*action)))

		action = torch.tanh(action)

		return action, log_p


	def loss(self, q, log_p, alpha):
		l = q - alpha*log_p
		#print(l.shape)
		return -(l).mean()

	def optimize(self, q, log_p, alpha=0.2):

	  torch.cuda.empty_cache()
	  self.optimizer.zero_grad()
	  loss = self.loss(q, log_p, alpha)
	  loss.backward(retain_graph=True)
	  self.optimizer.step()

	  return -loss.detach().numpy()

def main():
	pn = PolicyNetwork(in_dim=3, out_dim=1)
	x = torch.randn(10, 3)
	print(pn.forward(x))

if __name__ == "__main__":
	main()
