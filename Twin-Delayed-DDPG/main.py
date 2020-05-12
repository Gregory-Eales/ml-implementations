import gym
import torch
import numpy as np
from tqdm import tqdm

from Twin_Delayed_DDPG.twin_delayed_ddpg import DDPG

from matplotlib import pyplot as plt

def train(env, ddpg, epochs=10, episodes=200, steps=100, epsilon=0.3,
 render=False, graph=False, run=False):
	
	if graph: plt.ion()
	if render: env.render()

	avg_reward = []
	for e in tqdm(range(epochs)):

		

		reward = []
		for i_episode in range(episodes):

			s = env.reset()

			for t in range(steps):

				
				if e > 50:
					a = ddpg.act(s, epsilon=epsilon)

				if e == 50:
					ddpg.clear_buffer()

				else:
					a = ddpg.act(s, epsilon=0.3)

				s_p, r, d, info = env.step(a)
				if t==steps-1: d = True

				ddpg.store(s, a, r, s_p, d)
				s = s_p

				if d:
					ddpg.discount_reward(t)
					reward.append(r)
					break
		
		avg_reward.append(np.sum(reward)/len(reward))

		
		
		ddpg.update(iter=5, n=100)
		#run(ddpg, env, episodes=1, steps=200)

		#ddpg.clear_buffer()

		if graph:
			plt.title("Reward per Epoch")
			plt.xlabel("Epoch")
			plt.ylabel("Reward")

			
			plt.plot(np.linspace(0, len(ddpg.q_loss), num=len(avg_reward)).tolist(),
				np.array(avg_reward)/abs(np.max(avg_reward)), label="Reward")

			q_loss = np.array(ddpg.q_loss)
			q_loss = q_loss/q_loss.max()
			
			p_loss = np.array(ddpg.p_loss)/200

			plt.plot(q_loss, label="Q loss")
			plt.plot(p_loss, label="P loss")

			disc_reward = ddpg.buffer.discount_reward_buffer
			disc_reward = disc_reward/abs(np.max(disc_reward))

			#plt.plot(disc_reward, label="Discount Reward")

			plt.legend()
			plt.draw()
			plt.pause(0.0001)
			plt.clf()

		if run: run(ddpg, env, episodes=1, steps=200)
		if e%50 == 0: ddpg.clear_buffer()

	env.close()

	return avg_reward


def run(ddpg, env, episodes=10, steps=1000):

	env.render()

	for i_episode in range(episodes):

		s = env.reset()

		for t in range(steps):

			env.render()
			a = ddpg.act(s)
			s, r, d, info = env.step(a)
			if t==steps-1: d = True

			if d:
				pass
				#break

	env.close()


env = gym.make('Humanoid-v3')

s_size=env.observation_space.shape[0]
a_size=env.action_space.shape[0]

print("State Size:", s_size)
print("Action Size:", a_size)

ddpg = DDPG(in_dim=s_size, out_dim=a_size, p_alpha=1e-3, q_alpha=1e-3)
reward = train(env, ddpg, epochs=1000, episodes=1,
 steps=250, epsilon=0.1, render=False, graph=True, run=False)

#print(ddpg.p_loss)

run(ddpg, env, steps=1000)

plt.plot(reward/np.max(reward), label="Reward")
plt.plot(np.array(ddpg.q_loss)/np.max(ddpg.q_loss), label="Q loss")
plt.plot(np.array(ddpg.p_loss)/np.max(ddpg.p_loss), label="P loss")
plt.legend()
plt.show()
