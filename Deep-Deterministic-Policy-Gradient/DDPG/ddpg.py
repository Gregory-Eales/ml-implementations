import torch
from .buffer import Buffer
from .policy_network import PolicyNetwork
from .q_network import QNetwork


class DDPG(object):

    def __init__(self, in_dim, out_dim):

        self.buffer = Buffer()
        self.q_net = QNetwork(in_dim, out_dim)
        self.policy_network = PolicyNetwork(in_dim, out_dim, q_net=self.q_net)

        self.target_p = QNetwork(in_dim, out_dim)
        self.target_q = PolicyNetwork(in_dim, out_dim, q_net=self.target_p)

    def compute_targets(self):
    	pass

    def store(self, observation, reward, done):
    	self.buffer.store(observation, reward, done)

    def discount_reward(self, t):
    	reward = self.buffer.get_reward()[-t:]
    	disc_reward = []
    	for i in range(len(reward)):
    		r = 0
    		for j in range(len(reward)-i):
    			r += reward[i+j]*(0.99**j)
    		disc_reward.append(r)
    	self.buffer.store_disc_reward(disc_reward)

    def act(self, observation):
    	action = self.policy_network(observation)
    	action = (action + 2*torch.rand(1, 1) - 1).clamp(min=-1, max=1)
    	self.buffer.store_action(action)
    	return action.detach().numpy()

    def update(self):

    	self.update_q_net()
    	self.update_policy()

    