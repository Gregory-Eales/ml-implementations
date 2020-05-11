import gym
import torch

from DDPG.ddpg import DDPG

from matplotlib import pyplot as plt

def train(env, ddpg, epochs=10, episodes=200, steps=100, render=False):
    
    for e in range(epochs):

        for i_episode in range(episodes):

            s = env.reset()

            for t in range(steps):

                if render: env.render()
               
                a = ddpg.act(s)
                s_p, r, d, info = env.step(a)
                if t==steps-1: d = True

                ddpg.store(s, a, r, s_p, d)

                if d:
                    ddpg.discount_reward(t)
                s = s_p

        ddpg.update(iter=20)

    env.close()


env = gym.make('MountainCarContinuous-v0')
ddpg = DDPG(in_dim=2, out_dim=1)
train(env, ddpg, epochs=10, episodes=10, steps=100, render=False)

plt.plot(ddpg.buffer.discount_reward_buffer)
plt.show()
