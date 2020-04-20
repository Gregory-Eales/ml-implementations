import torch
from torch.nn import functional as F
from torch import optim
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer

class PolicyNetwork(pl.LightningModule):

    def __init__(self, in_dim, out_dim, q_net=None):

        super(PolicyNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.l1 = nn.Linear(in_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, out_dim)

        self.q_network = q_net

    def forward(self, x):
        
        out = torch.Tensor(x).reshape(-1, self.in_dim)
        out = self.l1(out)
        out = F.relu(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.l3(out)
        out = torch.tanh(out)

        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer, optim.lr_scheduler.StepLR(optimizer, step_size=1)

    def loss(self, output, target):
        loss = self.loss(output, target)

    def training_step(self, s_batch, batch_idx):
        actions = self.forward(s_batch)
        q = self.q_network(s_batch, actions).mean()
        self.logger.summary.scalar('loss', q)
        return q

def main():
    pn = PolicyNetwork(in_dim=3, out_dim=1)
    x = torch.ones(10, 3)
    print(pn.forward(x))

if __name__ == "__main__":
    main()
