import torch


class PolicyNetwork(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim, alpha):

        super(PolicyNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.alpha = alpha

        self.leaky_relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.l1 = torch.nn.Linear(in_dim, hid_dim)
        self.l2 = torch.nn.Linear(hid_dim, hid_dim)
        self.l3 = torch.nn.Linear(hid_dim, out_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.to(self.device)

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.l1(out)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.relu(out)
        return out.to(torch.device("cpu:0")).float()

    def loss(self):
        pass

    def optimize(self):
        pass

def main():
    pn = PolicyNetwork(3, 3, 3, 0.01)
    x = torch.ones(10, 3)
    print(pn.forward(x))

if __name__ == "__main__":
    main()
