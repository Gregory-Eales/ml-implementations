from ppo.ppo import PPO
import gym

torch.manual_seed(1)
np.random.seed(1)

env = gym.make('MountainCar-v0')
ppo = PPO(alpha=0.001, input_size=2, output_size=3)

ppo.train(env=env, epochs=1, steps=1000)
