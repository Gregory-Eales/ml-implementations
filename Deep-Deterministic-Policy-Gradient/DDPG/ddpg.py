import torch
from buffer import Buffer
from policy_network import PolicyNetwork
from q_network import QNetwork
from pytorch_lightning import Trainer


class DDPG(object):

	def __init__(self, in_dim, out_dim):

		self.buffer = Buffer()
		self.q_net = QNetwork(in_dim, out_dim)
		self.p_net = PolicyNetwork(in_dim, out_dim, q_net=self.q_net)

		self.target_q = QNetwork(in_dim, out_dim)
		self.target_p = PolicyNetwork(in_dim, out_dim, q_net=self.target_q)

		p_state_dict = self.p_net.state_dict()
		q_state_dict = self.q_net.state_dict()

		self.target_p.load_state_dict(p_state_dict)
		self.target_q.load_state_dict(q_state_dict)

		self.q_trainer = Trainer(logger=True, checkpoint_callback=True, early_stop_callback=False, callbacks=[], default_save_path=None, gradient_clip_val=0, process_position=0, num_nodes=1, gpus=None, num_tpu_cores=None, log_gpu_memory=None, progress_bar_refresh_rate=1, overfit_pct=0.0, track_grad_norm=-1, check_val_every_n_epoch=1, fast_dev_run=False, accumulate_grad_batches=1, max_epochs=1000, min_epochs=1, max_steps=None, min_steps=None, train_percent_check=1.0, val_percent_check=1.0, test_percent_check=1.0, val_check_interval=1.0, log_save_interval=100, row_log_interval=10, add_row_log_interval=None, distributed_backend=None, precision=32, print_nan_grads=False, weights_summary='full', weights_save_path=None, amp_level='O1', num_sanity_val_steps=5, truncated_bptt_steps=None, resume_from_checkpoint=None, profiler=None, benchmark=False, reload_dataloaders_every_epoch=False, gradient_clip=None, nb_gpu_nodes=None, max_nb_epochs=None, min_nb_epochs=None, use_amp=None, show_progress_bar=None, nb_sanity_val_steps=None)

	def compute_targets(self):
		o, a, r, d, t_b = self.get()

		print("o", o.shape)
		print("a", a.shape)
		print("r", r.shape)
		print("d_r", d.shape)
		print("t_b", t_b.shape)

		return d_r + 0.99*(1-d)*self.target_q.forward(o)

	def update_params(self, p=0.1):
		
		td = self.target_p.state_dict()
		md = self.p_net.state_dict()

		for name, param in td.items():
		    td[name].copy_(p*param + (1-p)*md[name])

		td = self.target_q.state_dict()
		md = self.q_net.state_dict()

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
		

	def act(self, state):
		action = self.p_net(state)
		action = (action + 2*torch.rand(1, 1) - 1).clamp(min=-1, max=1)
		return action.detach().numpy()

	def update(self):
		target = self.compute_targets()
		print("target", target.shape)
		self.update_q_net()
		self.update_policy()

	def update_q_net(self):
		pass

	def update_policy(self):
		pass

if __name__ == "__main__":

	ddpg = DDPG(2, 2)
	ddpg.update_params()

	

	