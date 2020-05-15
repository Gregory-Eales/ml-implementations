import gym
import torch
import numpy as np
from tqdm import tqdm
import time
from matplotlib import pyplot as plt


from sac.sac import SAC
from utils.utils import graph

def train(env, sac, epochs=10, episodes=200, steps=100,
 render=False, graphing=False, run=False):
	
	if graphing: plt.ion()
	if render: env.render()

	disc_r = []
	avg_reward = []
	for e in tqdm(range(epochs)):

		

		reward = []
		for i_episode in range(episodes):

			s = env.reset()

			for t in range(steps):

				
				a = sac.act(s)
				s_p, r, d, info = env.step(a)
				if t==steps-1: d = True

				sac.store(s, a, r, s_p, d)
				s = s_p

				if d:
					disc_r.append(sac.discount_reward(t))
					reward.append(r)
					break
		
		avg_reward.append(np.sum(reward)/len(reward))

		
		
		sac.update(iter=10, n=50)
	
		if graphing:
			graph(sac, avg_reward, disc_r)

		if run: run(sac, env, episodes=1, steps=200)
		if e%50 == 0: sac.clear_buffer()

	env.close()

	return avg_reward


def run(sac, env, episodes=10, steps=1000):

	env.render()

	for i_episode in range(episodes):

		s = env.reset()

		for t in range(steps):

			env.render()
			
			a = sac.act(s)
			
			s, r, d, info = env.step(a)
			if t==steps-1: d = True

			if d:
				pass
				#break

	env.close()


env = gym.make('LunarLanderContinuous-v2')


s_size=env.observation_space.shape[0]
a_size=env.action_space.shape[0]

print("State Size:", s_size)
print("Action Size:", a_size)

sac = SAC(in_dim=s_size, out_dim=a_size, p_alpha=1e-3, q_alpha=1e-3)
reward = train(env, sac, epochs=1000 , episodes=1,
 steps=200, render=False, graphing=True, run=False)

run(sac, env, episodes=1, steps=100)

"""
plt.plot(reward/np.max(reward), label="Reward")
plt.plot(np.array(sac.q_loss)/np.max(sac.q_loss), label="Q loss")
plt.plot(np.array(sac.p_loss)/np.max(sac.p_loss), label="P loss")
plt.legend()
plt.show()
"""
