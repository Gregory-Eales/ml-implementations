import torch
import numpy as np
import gym
import time
from matplotlib import pyplot as plt

from policy_network import PolicyNetwork
from value_network import ValueNetwork
from buffer import Buffer


class PPO(object):

    def __init__(self, alpha, in_dim, out_dim):

        self.out_dim = out_dim
        self.in_dim = in_dim

        self.buffer = Buffer(in_dim)
        self.value_net = ValueNetwork(alpha, in_dim, 1)
        self.policy_net = PolicyNetwork(alpha, in_dim, out_dim)
        self.old_policy_net = PolicyNetwork(alpha, in_dim, out_dim)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.historical_reward = []


    def store(self, state, action, reward):

        self.buffer.store_state(state)
        self.buffer.store_action(action)
        self.buffer.store_reward(reward)
        self.calculate_advantages()

    def calculate_advantages(self):

        s2 = torch.Tensor(self.buffer.states[-1])
        s1 = torch.Tensor(self.buffer.states[-2])
        r = self.buffer.rewards[-1]

        v = self.value_net(s1)
        q = self.value_net(s2)
        a = q-v+1

        self.buffer.store_advantage(a)

    def update(self, iter=50):
        states = self.buffer.get_states()
        disc_rewards = self.buffer.get_discounted_rewards()
        log_probs = self.buffer.get_log_probs()
        old_probs = self.buffer.get_old_log_probs()
        advantages = self.buffer.get_advantages()

        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.optimize(log_probs, old_probs, advantages, iter=1)

        self.value_net.optimize(states, disc_rewards, iter=iter)

    def get_action(self):
        state = torch.Tensor(self.buffer.states[-1]).reshape(1, self.in_dim)
        prediction = self.policy_net.forward(state)
        old_pred = self.old_policy_net.forward(state)

        action_probabilities = torch.distributions.Categorical(prediction)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)

        old_action_probabilities = torch.distributions.Categorical(old_pred)
        old_action = old_action_probabilities.sample()
        old_log_prob = old_action_probabilities.log_prob(old_action)

        self.buffer.store_log_prob(log_prob)
        self.buffer.store_old_log_probs(old_log_prob)

        # convert discrete to continuous2
        #action = -2.0 + 4/self.out_dim*action.float()



        return action.item()

    def train(self, env, n_steps, n_epoch, render=False, verbos=False):

        for i in range(n_epoch):

            for n in range(int(n_steps/200)):
                # reset env
                state = env.reset()
                self.buffer.store_state(state)
                for s in range(200):

                    if render: env.render()

                    # get action
                    action = self.get_action()

                    # get observation
                    state, reward, done, info = env.step(action)

                    # store metrics


                    if done:
                        self.buffer.store_action(action)
                        self.buffer.store_reward(reward)
                        self.buffer.discount_rewards(s)
                        self.calculate_advantages()
                        break

                    else: self.store(state, action, reward)

            # update networks
            self.update(iter=80)
            mean_reward = torch.sum(self.buffer.get_discounted_rewards().mean())
            print(mean_reward)
            self.historical_reward.append(mean_reward.item())

            self.buffer = Buffer(self.in_dim)

        plt.plot(self.historical_reward)
        plt.show()

def main():

    env = gym.make("CartPole-v0")
    ppo = PPO(alpha=0.001, in_dim=4, out_dim=2)
    ppo.train(env=env, n_steps=800, n_epoch=20, render=False)


if __name__ == "__main__":
    main()
