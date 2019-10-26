import torch


class ValueNetwork(torch.nn.Module):

    def __init__(self, input_size, alpha):

        super(ValueNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 1)

        self.relu = torch.nn.LeakyReLU()

        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())
        self.loss = torch.nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def normalize(self, x):
        x = np.array(x)
        x_mean = np.mean(x)
        x_std = np.std(x) if np.std(x) > 0 else 1
        x = (x-x_mean)/x_std
        return x

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out.to(torch.device('cpu:0'))

    def optimize(self, actions, observations, rewards, iter=iter):

        x = torch.cat([actions, observations], dim=1)

        for i in range(iter):
            torch.cuda.empty_cache()
            # zero the parameter gradients
            self.optimizer.zero_grad()

            self.loss = self.loss(x, rewards)
            # optimize
            self.loss.backward(retain_graph=True)

            self.optimizer.step()
