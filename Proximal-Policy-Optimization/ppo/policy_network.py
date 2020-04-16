import torch
from tqdm import tqdm

class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, in_dim, out_dim, epsilon=0.3):

        super(PolicyNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon
        self.define_network()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)
        self.prev_params = self.parameters()

    def define_network(self):
        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.l1 = torch.nn.Linear(self.in_dim, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, self.out_dim)

    def normalize(self):
        pass

    def loss(self, r_theta, advantages):
        clipped_r = r_theta.clamp(1.0 - self.epsilon, 1.0 + self.epsilon)
        return torch.min(r_theta*advantages, clipped_r).mean()

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.l1(out)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.sigmoid(out)

        return out.to(torch.device('cpu:0'))

    def optimize(self, adv, r, iter=1):

        num_batch = r.shape[0]//16
        rem_batch = r.shape[0]%16

        print("Training Policy Network: ")

        for b in tqdm(range(num_batch)):

            n1 = b*16
            n2 = (b+1)*16

            if b == 0:
                loss = self.loss(r[n1:n2], adv[n1:n2])
                loss.backward(retain_graph=True)
                self.optimizer.zero_grad()
                self.optimizer.step()

            else:
                loss = self.loss(r[n1:n2], adv[n1:n2])
                loss.backward(retain_graph=True)
                self.optimizer.zero_grad()
                self.optimizer.step()

        loss = self.loss(r[-rem_batch:], adv[-rem_batch:])
        loss.backward(retain_graph=True)
        self.optimizer.zero_grad()
        self.optimizer.step()



def main():

    t1 = torch.ones(1, 3)
    pn = PolicyNetwork(0.01, 3, 1)
    print(pn(t1))
    print(pn.parameters())


if __name__ == "__main__":
    main()
