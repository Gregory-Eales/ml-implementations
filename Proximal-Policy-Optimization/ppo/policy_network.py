import torch

class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, in_dim, out_dim):

        super(PolicyNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.l1 = torch.nn.Linear(in_dim, 128)
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.Linear(128, out_dim)


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    def define_network(self):


    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.sigmoid(out)

        return out.to(torch.device('cpu:0'))

def main():

    policy_net = PolicyNetwork(alpha=0.1, in_dim=3, in_dim=3)
