import torch
from tqdm import tqdm

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

        self.l1 = torch.nn.Linear(self.in_dim, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, self.out_dim)

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        return out.to(torch.device('cpu:0'))

    def optimize(self, iter, state, disc_reward):

        print("Training Value Network: ")

        print(state.shape)

        for i in tqdm(range(iter)):

            num_batch = state.shape[0]//16

            for b in range(num_batch):

                n1 = b*16
                n2 = (b+1)*16

                p = self.forward(state[n1:n2]).reshape(16, 1)
                loss = self.loss(p, disc_reward[n1:n2])
                loss.backward(retain_graph=True)
                self.optimizer.zero_grad()
                self.optimizer.step()


def main():

    t1 = torch.rand(100, 3)
    vn = ValueNetwork(0.01, 3, 1)
    vn.optimize(iter=100, state=t1, disc_reward=torch.rand(100, 1))



if __name__ == "__main__":
    main()
