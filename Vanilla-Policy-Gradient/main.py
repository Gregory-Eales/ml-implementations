import gym
from vpg import VPG

vpg = VPG()

env = gym.make('CartPole-v0')
for i_episode in range(1000):
    observation = env.reset()
    for t in range(10):
        env.render()
        print(observation)
        action = vpg.act()
        print("This is an action: ", action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
