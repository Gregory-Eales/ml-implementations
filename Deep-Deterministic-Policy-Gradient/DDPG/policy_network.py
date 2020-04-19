import torch
from torch.nn import functional as F
from torch import optim
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer

class PolicyNetwork(pl.LightningModule):

    def __init__(self, in_dim, out_dim):

        super(PolicyNetwork, self).__init__()

        self.l1 = nn.Linear(in_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, out_dim)


    def forward(self, x):
        
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.l3(out)
        out = F.relu(out)

        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer, optim.lr_scheduler.StepLR(optimizer, step_size=1)

    def loss(self, output, target):
        pass

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.loss(output, target)
        self.logger.summary.scalar('loss', loss)
        return loss

def main():
    pn = PolicyNetwork(3, 3, 3, 0.01)
    x = torch.ones(10, 3)
    print(pn.forward(x))

if __name__ == "__main__":
    main()
