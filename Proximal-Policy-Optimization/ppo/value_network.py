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
        self.to(self.device)


    def define_network(self):

        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        self.l1 = torch.nn.Linear(self.in_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, self.out_dim)

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        return out.to(torch.device('cpu:0'))

    def update(self, iter, state, disc_reward):
        for i in range(iter):
            p = self.forward(state)
            loss = self.loss(p, disc_reward)
            loss.backward(retain_graph=True)
            self.optimizer.zero_grad()
            self.optimizer.step()


    def optimize(self, states, rewards, iter):
        pass

def main():

    t1 = torch.ones(1, 3)
    vn = ValueNetwork(0.01, 3, 1)
    print(vn(t1))



if __name__ == "__main__":
    main()
