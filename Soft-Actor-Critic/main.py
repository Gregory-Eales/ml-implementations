import gym
import torch
import numpy as np
from tqdm import tqdm
import time

from sac.sac import SAC

from matplotlib import pyplot as plt

def train(env, sac, epochs=10, episodes=200, steps=100,
 render=False, graph=False, run=False):
	
	if graph: plt.ion()
	if render: env.render()

	disc_r = []
	avg_reward = []
	for e in tqdm(range(epochs)):

		

		reward = []
		for i_episode in range(episodes):

			s = env.reset()

			for t in range(steps):

				
				start_time = time.time()
				a = sac.act(s)
				print(time.time()-start_time)

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
	
		if graph:
			plt.title("Reward per Epoch")
			plt.xlabel("Epoch")
			plt.ylabel("Reward")

			
			ls1 = np.linspace(0, len(sac.q1_loss), num=len(avg_reward)).tolist()
			avg_rp = np.array(avg_reward)
			plt.plot(ls1, avg_rp/np.max(np.abs(avg_rp)), label="Reward")

			q1_loss = np.array(sac.q1_loss)
			q1_loss = q1_loss/q1_loss.max()

			q2_loss = np.array(sac.q2_loss)
			q2_loss = q2_loss/q2_loss.max()
			
			p_loss = np.array(sac.p_loss)
			p_loss = p_loss/abs(np.max(np.abs(p_loss)))

			plt.plot(q1_loss, label="Q1 loss")
			plt.plot(q2_loss, label="Q2 loss")

			ls2 = np.linspace(0, len(sac.q1_loss), num=len(p_loss)).tolist()
			plt.plot(ls2, p_loss, label="P loss")

			disc_rp = np.array(disc_r)
			disc_rp = disc_rp/np.max(np.abs(disc_rp))
			ls3 = np.linspace(0, len(sac.q1_loss), num=len(disc_rp)).tolist()
			
			plt.plot(ls3, disc_rp, label="Discount Reward")

			plt.legend()
			plt.draw()
			plt.pause(0.0001)
			plt.clf()

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
			for i in range(20):
				a = sac.act(s)
			print(a)
			s, r, d, info = env.step(a)
			if t==steps-1: d = True

			if d:
				pass
				#break

	env.close()


env = gym.make('LunarLander-v2')

"""
s_size=env.observation_space.shape[0]
a_size=env.action_space.shape[0]

print("State Size:", s_size)
print("Action Size:", a_size)
"""
sac = SAC(in_dim=8, out_dim=4, p_alpha=1e-3, q_alpha=1e-3)
#reward = train(env, sac, epochs=1 , episodes=1,
# steps=200, render=False, graph=True, run=False)

run(sac, env, episodes=3, steps=100)

"""
plt.plot(reward/np.max(reward), label="Reward")
plt.plot(np.array(sac.q_loss)/np.max(sac.q_loss), label="Q loss")
plt.plot(np.array(sac.p_loss)/np.max(sac.p_loss), label="P loss")
plt.legend()
plt.show()
"""
