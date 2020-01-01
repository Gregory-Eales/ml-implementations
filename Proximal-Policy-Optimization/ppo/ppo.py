import torch
from tqdm import tqdm


from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from .buffer import Buffer



class PPO(object):


    def __init__(self):

        self.value_network = ValueNetwork()
        self.policy_network = PolicyNetwork()
        self.old_policy_network = PolicyNetwork()
        state_dict = self.policy_network.state_dict()
        self.old_policy_network.load_state_dict(state_dict)
        self.buffer = Buffer()
        self.mean_reward()

    def act(self, state):

        print(state)
        pass

    def calculate_advantages(self):
        pass


    def update(self, epochs=80):

        states, policy, old_policy, rewards, advantages = self.buffer.get()

        r = policy/old_policy

        self.policy_network.optimize(advantages, r)
        self.value_network.optimize(states, rewards)

    def train(self, env, n_steps, n_epochs):

        for epoch in range(n_epochs):

            state = env.reset()

            print("Collecting Trajectories")
            for step in tqdm(range(n_steps)):

                action = self.act(state)

                state, reward, done, info = env.step(action)

                self.buffer.store(state, action, reward)

                if done:

                    self.calculate_advantages()


            mean_reward = self.buffer.get_rewards()
            mean_reward = torch.sum(mean_reward)/mean_reward.shape[0]

            if self.mean_reward[-1] > 150:
                file_name = "policy_params.pt"
                torch.save(self.policy_network.state_dict(), file_name)

    def play(self):
        pas
