from ppo.ppo import PPO
import gym
import torch
import numpy as np
from matplotlib import pyplot as plt

torch.manual_seed(1)
np.random.seed(1)

env = gym.make('CartPole-v0')
ppo = PPO(alpha=0.001, in_dim=4, out_dim=2)

ppo.train(env=env, n_epochs=10, n_steps=500)


plt.plot(ppo.mean_reward)
plt.show()
