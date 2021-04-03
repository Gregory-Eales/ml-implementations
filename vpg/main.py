# import dependencies
import gym
from matplotlib import pyplot as plt

# import vpg algorithm
from vpg.vpg import VPG
from utils import plathrough

# initialize vpg algorithm object
vpg = VPG()

# initialize environment
env = gym.make('CartPole-v0')

# loop through episodes
for i_episode in range(1000):

    # reset environment
    observation = env.reset()

    # take n time steps for each episode
    for t in range(10):

        # render env screen
        env.render()

        # print observation
        print(observation)

        # get action
        action = vpg.act()

        # print action taken by vpg agent
        print("This is an action: ", action)

        # get state + reward
        observation, reward, done, info = env.step(action)

        # check if episode is terminal
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

# close environment
env.close()
