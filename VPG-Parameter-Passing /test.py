import gym
import torch
import numpy as np

from vpg.vpg import VPG


torch.manual_seed(1)
np.random.seed(1)

env = gym.make('CartPole-v0')

vpg = VPG(alpha=0.001, input_dims=4, output_dims=2)

vpg.policy_network.load_state_dict(torch.load("policy_params.pt"))

vpg.play(env)
