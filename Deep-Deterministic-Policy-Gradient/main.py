import gym
import torch

from DDPG.ddpg import DDPG

from matplotlib import pyplot as plt

def train(env, ddpg, epochs=10, episodes=100, steps=100, render=False):
    
    for e in range(epochs):

        for i_episode in range(episodes):

            observation = env.reset()

            for t in range(steps):

                if render: env.render()
               
                action = ddpg.act(observation)
                observation, reward, done, info = env.step(action)
                if t==steps-1: done = True
                ddpg.store(observation, reward, done)

                if done:
                    dr = ddpg.discount_reward(t)
                    break

        ddpg.update()

    env.close()


env = gym.make('MountainCarContinuous-v0')
ddpg = DDPG(in_dim=2, out_dim=1)
train(env, ddpg, epochs=1, episodes=10, steps=100, render=False)


