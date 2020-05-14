import torch
from torch.nn import functional as F
from torch import optim
from torch import nn

class QNetwork(torch.nn.Module):

		def __init__(self, in_dim, out_dim, alpha=0.01):

				super(QNetwork, self).__init__()

				self.l1 = nn.Linear(in_dim, 128)
				self.l2 = nn.Linear(128, 128)
				self.l3 = nn.Linear(128, 64)
				self.l4 = nn.Linear(64, out_dim)
				self.relu = nn.LeakyReLU()


				self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())

			
				self.loss = torch.nn.MSELoss()

				self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
				self.to(self.device)


		def forward(self, s, a):

				out = torch.cat([s, a], dim=1)
				out = torch.Tensor(out).to(self.device)

				out = self.l1(out)
				out = self.relu(out)
				out = self.l2(out)
				out = self.relu(out)
				out = self.l3(out)
				out = self.relu(out)
				out = self.l4(out)
				out = self.relu(out)

				#out = torch.clamp(out, -200, 200)
				

				return out.to(torch.device("cpu:0"))

		def optimize(self, s, a, y):

			q = self.forward(s, a)
			torch.cuda.empty_cache()
			self.optimizer.zero_grad()
			loss = self.loss(q, y)
			loss.backward(retain_graph=True)
			self.optimizer.step()

			return loss.detach().numpy()


def main():

	qnet = QNetwork(in_dim=3, out_dim=1)
	trainer = Trainer(max_epochs=1)
	trainer.fit(qnet)






