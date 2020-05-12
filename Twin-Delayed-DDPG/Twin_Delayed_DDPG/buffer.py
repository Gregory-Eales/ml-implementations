import torch
import numpy


class Buffer(object):

	def __init__(self):
		
		self.reset()

	def reset(self):
		self.state_buffer = []
		self.state_prime_buffer = []
		self.reward_buffer = []
		self.discount_reward_buffer = []
		self.action_buffer = []
		self.policy_buffer = []
		self.advantage_buffer = []
		self.terminal_buffer = []


	def store_terminal(self, done):
		if done: self.terminal_buffer.append(1)
		else: self.terminal_buffer.append(0)

	def store(self, s, a, r, s_p, d):
		self.store_state(s)
		self.store_action(a)
		self.store_reward(r)
		self.store_state_prime(s_p)
		self.store_terminal(d)

	def store_state(self, state):
		self.state_buffer.append(state.reshape(-1))

	def store_state_prime(self, s_p):
		self.state_prime_buffer.append(s_p.reshape(-1))

	def store_reward(self, r):
		self.reward_buffer.append(r)

	def store_disc_reward(self, disc_r):
		self.discount_reward_buffer += disc_r

	def store_action(self, act):
		self.action_buffer.append(act)

	def store_policy(self, p):
		self.policy_buffer.append(p)

	def store_advantage(self, adv):
		self.advantage_buffer.append(adv)


	def random_sample(self, s, a, r, s_p, d, n=50):

		rand_perm = torch.randperm(s.shape[0])
		[rand_perm][0:n]

		s = s[rand_perm][0:n]
		a = a[rand_perm][0:n]
		r = r[rand_perm][0:n]
		s_p = s_p[rand_perm][0:n]
		d = d[rand_perm][0:n]

		return s, a, r, s_p, d

	def get(self):

		s = torch.Tensor(self.state_buffer)
		a = torch.Tensor(self.action_buffer).reshape(-1, 2)
		r = torch.Tensor(self.reward_buffer).reshape(-1, 1)
		s_p = torch.Tensor(self.state_prime_buffer)
		d = torch.Tensor(self.terminal_buffer).reshape(-1, 1)

		#print(s.shape, a.shape, r.shape, s_p.shape, d.shape)

		return s, a, r, s_p, d

	def get_state(self):
		return self.state_buffer

	def get_state_prime(self):
		return self.state_prime_buffer

	def get_reward(self):
		return self.reward_buffer

	def get_disc_reward(self):
		return self.discount_reward_buffer

	def get_action(self):
		return self.action_buffer

	def get_policy(self):
		return self.policy_buffer

	def get_advantage(self):
		return self.advantage_buffer


