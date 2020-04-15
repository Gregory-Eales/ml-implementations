import torch
from tqdm import tqdm
import numpy as np

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from .buffer import Buffer

class PPO(object):

    def __init__(self, alpha=0.001, in_dim=3, out_dim=2):

        self.value_network = ValueNetwork(alpha=alpha, in_dim=in_dim, out_dim=1)
        self.policy_network = PolicyNetwork(alpha=alpha, in_dim=in_dim, out_dim=out_dim)
        self.old_policy_network = PolicyNetwork(alpha=alpha, in_dim=in_dim, out_dim=out_dim)
        state_dict = self.policy_network.state_dict()
        self.old_policy_network.load_state_dict(state_dict)
        self.buffer = Buffer()
        self.mean_reward = None

    def act(self, state):

        # convert to torch get_tensors
        s = torch.tensor(state).reshape(-1, len(state)).float()

		# get policy prob distrabution
        prediction = self.policy_network.forward(s)

		# get action probabilities
        action_probabilities = torch.distributions.Categorical(prediction)

		# sample action
        action = action_probabilities.sample()

        log_prob = action_probabilities.log_prob(action)

        return action.item()

    def calculate_advantage(self):

        states = self.buffer.get_states()

        v2 = self.value_network.forward(states[-1])
        v1 = self.value_network.forward(states[-2])

        return 1 + v2 - v1

    def discount_reward(self):

        r = np.array(self.buffer.get_rewards())
        disc_r = []
        discount = 0.99

        for i in reversed(range(r.shape[0])):

            disc_r.append((discount**i)*r+np.sum(disc_r))


    def update(self, epochs=80):

        states, policy, old_policy, rewards, advantages = self.buffer.get()

        r = policy/old_policy

        self.policy_network.optimize(advantages, r)
        self.value_network.optimize(states, rewards, epochs=epochs)

    def train(self, env, n_steps, n_epochs):

        for epoch in range(n_epochs):

            state = env.reset()

            self.buffer.store_state(state)

            print("Collecting Trajectories")
            for step in tqdm(range(n_steps)):

                action = self.act(state)

                state, reward, done, info = env.step(action)

                self.buffer.store_trajectory(state, action, reward)


                self.calculate_advantage()

                if done:
                    self.discount_reward()

            self.update(epoch=80)

            mean_reward = self.buffer.get_rewards()
            mean_reward = torch.sum(mean_reward)/mean_reward.shape[0]

            if self.mean_reward[-1] > 150:
                file_name = "policy_params.pt"
                torch.save(self.policy_network.state_dict(), file_name)

    def play(self):
        pas


def main():

    import gym
    torch.manual_seed(1)
    np.random.seed(1)

    env = gym.make('CartPole-v0')
    ppo = PPO(alpha=0.001, input_dims=3, output_dims=2)
    ppo.train(env, n_epoch=1000, n_steps=800, render=False, verbos=False)




if __name__ == "__main__":
    main()
