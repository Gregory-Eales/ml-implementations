import torch

class VPG(object):

	def __init__(self, observation_size, action_size, size, gamma=0.99, lam=0.95):

		######################
		# initialize buffers #
		######################

		# buffer of observation inputs
		self.observation_buffer = torch.zeros(observation_size)

		# buffer of outputed actions
		self.action_buffer = torch.zeros(action_size)

		# buffer of output action advantages over other actions
		self.advantage_buffer = torch.zeros(size, dtype=torch.float32)

		# buffer of returned rewards
		self.reward_buffer = torch.zeros(size, dtype=torch.float32)

		# buffer of cumilative reward aka return
		self.return_buffer = torch.zeros(size, dtype=torch.float32)

		# buffer of expected return in a state with an action
		self.value_buffer = torch.zeros(size, dtype=torch.float32)

		# buffer of log p values
		self.logp_buffer = torch.zeros(size, dtype=torch.float32)

		# gamma and lam values
		self.gamma, self.lam = gamma, lam

	def store_trajectory(self, observation, action, reward, value, logp):
		pass

	def finish_path(self, last_val=0):
		pass

	def get_buffer_data(self):
		pass
