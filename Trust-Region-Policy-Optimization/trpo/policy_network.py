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

    def kl_divergence(self, old_pi, pi, advantage):
        return torch.sum(old_pi*torch.log(old_pi/pi), 1).mean()

    def hessian_conjugate(self):
        pass

    def loss(self, actions, advantages, prev_params):

        g = torch.sum(actions*advantages)/actions.shape[0]
        delta = self.kl_divergence()
        x = self.hessian_conjugate()

        loss = 2*delta*x/(x.T*self.hessian_conjugate(x))
        loss = torch.sqrt(loss)

        return loss


    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        out = out.to(torch.device('cpu:0'))
        return out

    def optimize(self, log_probs, old_probs, advantages):
        self.optimizer.zero_grad()
        loss = self.loss(actions, advantage, prev_params)
        loss.backward(retain_graph=True)
        self.optimizer.step()
