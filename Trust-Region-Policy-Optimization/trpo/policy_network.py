import torch
import numpy as np


class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, input_size, output_size):

        super(PolicyNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_size)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def loss(self, log_prob, advantage):
        pass

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        out = out.to(torch.device('cpu:0'))

    def optimize(self, iterations=1):
        pass
