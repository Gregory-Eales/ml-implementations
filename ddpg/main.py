import gym
import torch
import numpy as np
from tqdm import tqdm
from copy import copy

from DDPG.ddpg import DDPG

from matplotlib import pyplot as plt

def train(env, ddpg, epochs=10, episodes=200, steps=100, render=False, graph=False):
    
    if graph: plt.ion()

    avg_reward = []
    hist_adr = []
    avg_disc_reward = 0
    for e in tqdm(range(epochs)):
        reward = []
        disc_reward = []
        for i_episode in range(episodes):

            s = env.reset()

            for t in range(steps):

                if render: env.render()

                a = ddpg.act(s)

                s_p, r, d, info = env.step(a)
                if t==steps-1: d = True

                ddpg.store(s, a, r, s_p, d)
                s = s_p

                if d:
                    
                    reward.append(r)
                    disc_reward.append(ddpg.discount_reward(t))
                    break
        
        avg_reward.append(np.sum(reward)/len(reward))
        avg_disc_reward+=np.sum(disc_reward)/len(disc_reward)
        hist_adr.append(copy(avg_disc_reward))
        
        
        ddpg.update(iter=10, n=100)
        #run(ddpg, env, episodes=1, steps=200)

        #ddpg.clear_buffer()

        if graph:
            plt.title("Reward per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Reward")

            
            #plt.plot(np.linspace(0, len(ddpg.q_loss), num=len(avg_reward)).tolist(),
                #np.array(avg_reward)/abs(np.max(avg_reward)), label="Reward")

            q_loss = np.array(ddpg.q_loss)
            q_loss = q_loss/q_loss.max()
            
            p_loss = np.array(ddpg.p_loss)/2000

            plt.plot(q_loss, label="Q loss")
            plt.plot(p_loss, label="P loss")

            ls = np.linspace(0, len(ddpg.q_loss), num=len(hist_adr))
            ha = np.array(hist_adr)/abs(np.max(np.abs(hist_adr)))
            plt.plot(ls,ha,label="Discounted Reward")
            plt.legend()
            plt.draw()
            
            
            plt.pause(0.0001)
            plt.clf()

        #run(ddpg, env, episodes=1, steps=200)

    env.close()

    return avg_reward


def run(ddpg, env, episodes=10, steps=200):

    for i_episode in range(episodes):

        s = env.reset()

        for t in range(steps):

            env.render()
            a = ddpg.act(s)
            s, r, d, info = env.step(a)
            if t==steps-1: d = True

            if d:
                break

    env.close()


env = gym.make('LunarLanderContinuous-v2')
ddpg = DDPG(in_dim=8, out_dim=2, p_alpha=1e-3, q_alpha=1e-3)
reward = train(env, ddpg, epochs=1000, episodes=1,
 steps=200, render=False, graph=True)

#print(ddpg.p_loss)

run(ddpg, env)

plt.plot(reward/np.max(reward), label="Reward")
plt.plot(np.array(ddpg.q_loss)/np.max(ddpg.q_loss), label="Q loss")
plt.plot(np.array(ddpg.p_loss)/np.max(ddpg.p_loss), label="P loss")
plt.legend()
plt.show()
