import torch

class ValueNetwork(torch.nn.Module):

    def __init__(self, alpha, in_dim, out_dim):

        super(ValueNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.define_network()

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha)
        self.loss = torch.nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu:0')

    def define_network(self):

        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        self.l1 = torch.nn.Linear(self.in_dim, 128)
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.Linear(128, self.out_dim)

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.sigmoid(out)
        return out.to(torch.device('cpu:0'))

    def update(self):
        pass

def main():

    t1 = torch.ones(1, 3)
    vn = ValueNetwork(0.01, 3, 1)
    print(vn(t1))



if __name__ == "__main__":
    main()
