import torch
from torch.distributions.normal import Normal

from .buffer import Buffer
from .policy_network import PolicyNetwork
from .q_network import QNetwork



class SAC(object):

	def __init__(self, in_dim, out_dim, p_alpha=1e-10, q_alpha=1e-10):

		self.out_dim = out_dim
		self.in_dim = in_dim

		self.buffer = Buffer()
		self.q1_net = QNetwork(in_dim+out_dim, 1, alpha=q_alpha)
		self.q2_net = QNetwork(in_dim+out_dim, 1, alpha=q_alpha)
		self.p_net = PolicyNetwork(in_dim, out_dim, alpha=p_alpha)

		self.target_q1 = QNetwork(in_dim+out_dim, 1)
		self.target_q2 = QNetwork(in_dim+out_dim, 1)

		q1_state_dict = self.q1_net.state_dict()
		q2_state_dict = self.q2_net.state_dict()

		self.target_q1.load_state_dict(q1_state_dict)
		self.target_q2.load_state_dict(q2_state_dict)

		self.q1_loss = []
		self.q2_loss = []
		self.p_loss = []
		
	def compute_targets(self, r, s_p, d, alpha=0.3):

		samp_action, log_p = self.act_p(s_p)

		targ1 = self.target_q1.forward(s_p, samp_action)
		targ2 = self.target_q2.forward(s_p, samp_action)

		targ = torch.min(targ1, targ2)
		
		return r + (1-d)*(targ-alpha*log_p)

	def update_params(self, p=0.995):

		td = self.target_q1.state_dict()
		md = self.q1_net.state_dict()

		for name, param in td.items():
			td[name].copy_(p*param + (1-p)*md[name])

		td = self.target_q2.state_dict()
		md = self.q2_net.state_dict()

		for name, param in td.items():
			td[name].copy_(p*param + (1-p)*md[name])

	def store(self, s, a, r, s_p, d):
		self.buffer.store(s, a, r, s_p, d)

	def get(self):
		return self.buffer.get()

	def discount_reward(self, t):
		reward = self.buffer.get_reward()[-t-1:]
		disc_reward = []
		for i in range(len(reward)):
			r = 0
			for j in range(len(reward)-i):
				r += reward[i+j]*(0.99**j)
			disc_reward.append(r)
		self.buffer.store_disc_reward(disc_reward)
		return  disc_reward[1]

	def act(self, state):

		prediction = self.p_net.forward(state)

		return prediction.detach().numpy()[0]

	def act_p(self, state):

		p, log_p = self.p_net.log_forward(state)

		return p, log_p


	def update(self, iter=10, n=100, alpha=0.4):

		s, a, r, s_p, d = self.buffer.get()

		for i in range(iter):
			samp_s, samp_a, samp_r, samp_s_p, samp_d = self.buffer.random_sample(s, a, r, s_p, d, n=n)

			y = self.compute_targets(samp_r, samp_s_p, samp_d, alpha=alpha)

			q1_loss, q2_loss = self.update_q_net(samp_s, samp_a, y)
			

			self.q1_loss.append(q1_loss)
			self.q2_loss.append(q2_loss)

			
			p_loss = self.update_p_net(samp_s, y, samp_a)
			self.p_loss.append(p_loss)
			self.update_params()

	def update_q_net(self, s, a, y):
		q1_loss = self.q1_net.optimize(s, a, y)
		q2_loss = self.q2_net.optimize(s, a, y)

		return q1_loss, q2_loss

	def update_p_net(self, s, y, l_p, alpha=0.4):
		

		action, log_p = self.act_p(s)


		"""
		print("###################")
		print(log_p)
		print("###################")
		print(action)
		print("###################")
		"""

		q1 = self.target_q1.forward(s, action)
		q2 = self.target_q2.forward(s, action)

		q = torch.min(q1, q2)
		
		return self.p_net.optimize(q, log_p, alpha=alpha)

	def clear_buffer(self):
		self.buffer = Buffer()



if __name__ == "__main__":

	sac = SAC(2, 2)
	SAC.update_params()

	

	