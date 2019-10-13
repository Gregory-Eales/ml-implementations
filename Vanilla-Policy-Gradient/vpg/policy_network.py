import torch
import numpy as np


class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, input_dims, output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims

        # inherit from nn module class
        super(PolicyNetwork, self).__init__()

        # initialize_network
        self.initialize_network()

        # define optimizer
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())

        # get device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        print(torch.cuda.is_available())
        print(self.device)
        self.to(self.device)

    def loss(self, log_probs, advantages):
        loss = -log_probs*advantages
        loss = torch.sum(loss)
        return loss

    # initialize network
    def initialize_network(self):

		# define network components
        self.fc1 = torch.nn.Linear(self.input_dims, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, self.output_dims)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out.to(torch.device('cpu:0'))

    def normalize(self, x):
        x = np.array(x)
        x_mean = np.mean(x)
        x_std = np.std(x) if np.std(x) > 0 else 1
        x = (x-x_mean)/x_std
        return x

    def update(self, actions, advantages, iter):

        advantages = self.normalize(advantages)

        actions = torch.Tensor(actions).to(self.device).reshape(-1, 1)
        advantages = torch.Tensor(advantages).to(self.device)

        # calculate loss
        loss = self.loss(actions, advantages)

        for i in range(iter):

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # optimize
            loss.backward(retain_graph=True)
            self.optimizer.step()


def main():

    x = torch.ones(100, 2, requires_grad=True)
    y = torch.ones(100, 1, requires_grad=True)/10

    # define PolicyNetwork
    pn = PolicyNetwork(alpha=0.1, input_dims=2, output_dims=1)

    pn.update(x, y, 100)

if __name__ == "__main__":
    main()
