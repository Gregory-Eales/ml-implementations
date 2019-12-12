import torch
from tools import load_mp4_cv2

class DeepFakeDiscriminator(torch.nn.Module):

    def __init__(self, alpha=0.01):

        super(DeepFakeDiscriminator, self).__init__()

        self.initialize_parameters()

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha)
        self.loss = torch.nn.CrossEntropyLoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.to(self.device)

    def initialize_parameters(self):

        self.conv1 = torch.nn.Conv3d(3, 16, [10, 40, 40])
        self.conv2 = torch.nn.Conv3d(128, 32, [10, 40, 40])
        self.conv3 = torch.nn.Conv3d(256, 64, [10, 40, 40])

        self.l1 = torch.nn.Linear(in_features=64, out_features=64)
        self.l2 = torch.nn.Linear(in_features=64, out_features=1)

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


def main():
    dfd = DeepFakeDiscriminator()
    p1 = "/Users/gregeales/Desktop/Repositories/ML-Reimplementations/"
    p2 = "Deep-Fake-Discriminator/DFD/sample_vids/vid1.mp4"
    print("Initialized Net")
    x = load_mp4_cv2(p1+p2)
    x = torch.Tensor(x)
    print("Loaded Tensor")
    x = x.reshape(1, 3, 10, 1080, 1920)
    print(x.shape)
    print(dfd.forward(x).shape)



if __name__ == "__main__":
    main()
