import torch
from tqdm import tqdm


from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from .buffer import Buffer



class PPO(object):


    def __init__(self):

        self.value_network = ValueNetwork()
        self.policy_network = PolicyNetwork()
        self.buffer = Buffer()

    def act(self, state):

        print(state)
        pass

    def calculate_advantages(self):
        pass

    def update(self, epochs=80):

        states, actions, rewards = self.buffer.get()

        self.value_network.optimize()
        self.policy_network.optimize()

    def train(self, env, n_steps, n_epochs):

        for epoch in range(n_epochs):

            state = env.reset()
            for step in tqdm(range(n_steps)):

                action = self.act(state)

                state, reward, done, _  = env.step(action)

                self.buffer.store(state, action, reward)

                if done:
                    pass








    def play(self):
        pas
