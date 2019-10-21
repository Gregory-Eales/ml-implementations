import torch


class PolicyNetwork(torch.nn.Module):

    def __init__(self, input_size, output_size, alpha):

        super(PolicyNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_size)

        self.relu = torch.nn.LeakyReLU()

        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def normalize(self, x):
        x = np.array(x)
        x_mean = np.mean(x)
        x_std = np.std(x) if np.std(x) > 0 else 1
        x = (x-x_mean)/x_std
        return x

    def loss(self):
        pass

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out.to(torch.device('cpu:0'))

    def optimize(self, actions, rewards):
        pass
