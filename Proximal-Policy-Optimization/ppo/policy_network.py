import torch

class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, in_dim, out_dim, epsilon=0.2):

        super(PolicyNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon
        self.define_network()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

        self.prev_params = self.parameters()

    def define_network(self):
        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.l1 = torch.nn.Linear(self.in_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, self.out_dim)

    def normalize(self):
        pass

    def loss(self, log_probs, old_log_probs, advantages):
        r_theta = log_probs/old_log_probs
        clipped_r = r_theta.clamp(1.0 - self.epsilon, 1.0 + self.epsilon)
        return -torch.min(r_theta*advantages, clipped_r).mean()

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.l1(out)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.relu(out)

        return out.to(torch.device('cpu:0'))

    def optimize(self, log_probs, old_log_probs, advantages, iter=10):

        loss = self.loss(log_probs, old_log_probs, advantages)
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
