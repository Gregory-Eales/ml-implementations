from ppo.ppo import PPO
import gym
import torch
import numpy as np

torch.manual_seed(1)
np.random.seed(1)

env = gym.make('Pendulum-v0')
ppo = PPO(alpha=0.001, in_dim=2, out_dim=3)

ppo.train(env=env, epochs=1, steps=1000)
