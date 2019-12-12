import torch

class DeepFakeDiscriminator(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size):

        super(DeepFakeDiscriminator, self).__init__()

        self.initialize_parameters()

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha)
        self.loss = torch.nn.CrossEntropyLoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.to(self.device)

    def initialize_parameters(self):

        self.conv1 = torch.nn.Conv3d(in_channels, 128, 40)
        self.conv2 = torch.nn.Conv3d(128, 256, 40)
        self.conv3 = torch.nn.Conv3d(256, 512, 40)

        self.l1 = torch.nn.Linear(in_features=512, out_features=512)
        self.l2 = torch.nn.Linear(in_features=512, out_features=1)

        self.sigmoid = torch.nn.Sigmoid()
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.l1(out)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.sigmoid(out)
        return out.to(torch.device("cpu:0"))

    def optimize(self, x, y, iter=10):

        for i in range(iter):
            pass

main():
    dfd = DeepFakeDiscriminator()
