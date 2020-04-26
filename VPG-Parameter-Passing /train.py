import gym
import torch
import numpy as np

import gym
from matplotlib import pyplot as plt

# import vpg algorithm
from vpg2p.vpg import VPG
#from utils import playthrough

torch.manual_seed(1)
np.random.seed(1)

env = gym.make('CartPole-v0')

vpg = VPG(alpha=0.001, input_dims=4, output_dims=2)

vpg.train(env, n_epoch=1000, n_steps=800, render=False, verbos=False)