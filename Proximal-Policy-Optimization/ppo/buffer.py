import torch
import numpy as np

class Buffer(object):

    def __init__(self):

        self.rewards = []
        self.states = []
        self.actions = []
        self.predictions = []
        self.old_log_probs = []
        self.discounted_rewards = []
        self.advantages = []
        self.log_probs = []

    def store_reward(self, reward):
        self.rewards.append(reward.tolist())

    def store_state(self, state):
        # in: numpy array
        self.states.append(state.reshape([1, 3]).tolist()[0])

    def store_action(self, action):
        self.actions.append(action)

    def store_prediction(self, prediction):
        self.predictions.append(prediction)

    def store_old_log_probs(self, old_log_prob):
        self.old_log_probs.append(old_log_prob)

    def store_log_prob(self, lg_prob):
        self.log_probs.append(lg_prob)

    def store_advantage(self, adv):
        self.advantages.append(adv)

    def discount_rewards(self, n_steps):

        r  = self.rewards[-n_steps:]
        r_discounted = []
        prev_val = 0
        for i in reversed(range(n_steps)):
            r_discounted.append(r[i]*(0.9**(i)))

        r_discounted = list(reversed(r_discounted))

        for i in range(n_steps-1):
            r_discounted[i] += sum(r_discounted[i+1:])


        self.discounted_rewards += r_discounted

    def get_rewards(self):
        return self.rewards

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_discounted_rewards(self):
        return self.discount_rewards

    def get_log_probs(self):
        return self.log_probs

    def get_old_log_probs(self):
        return self.old_log_probs

    def get_data(self):
        states = torch.Tensor(self.states)
        actions = torch.Tensor(self.actions)
        rewards = torch.Tensor(self.rewards)
        disc_reward = torch.Tensor(self.discounted_rewards)
        advantage = torch.cat(self.advantages)

        return states, actions, rewards, disc_reward, advantage
