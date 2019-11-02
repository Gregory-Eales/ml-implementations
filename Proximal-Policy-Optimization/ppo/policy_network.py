import torch

class PolicyNetwork(torch.nn.Module):

    def __init__(self, input_size, output_size, alpha):

        super(PolicyNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_size)

        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def normalize(self, x):
        x = np.array(x)
        x_mean = np.mean(x)
        x_std = np.std(x) if np.std(x) > 0 else 1
        x = (x-x_mean)/x_std
        return x

    def min(self, old_policy, policy, advantage, epsilon):
        print("old p", old_policy.shape)
        print("p", policy.shape)
        p = torch.sum(old_policy/policy, dim=0)*advantage
        advantage[advantage>=0] *= (1+epsilon)
        advantage[advantage<0] *= (1-epsilon)
        a = torch.sum(advantage)
        p = torch.sum(p)
        if p > a: return a
        else: return p

    def loss(self, old_policy, policy, advantage, epsilon):
        p = self.min(old_policy, policy, advantage, epsilon)
        return torch.sum(p)/(policy.shape[0])

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out.to(torch.device('cpu:0'))

    def optimize(self, old_policy, policy, advantage, epsilon):

        torch.cuda.empty_cache()
        # zero the parameter gradients
        self.optimizer.zero_grad()

        self.loss = self.loss(old_policy, policy, advantage, epsilon)
        # optimize
        self.loss.backward(retain_graph=True)

        self.optimizer.step()
