import torch

class PolicyNetwork(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(PolicyNetwork, self).__init__()


        self.in_dim = in_dim
        self.out_dim = out_dim
        self.optimizer = torch.optim.Adam(params=self.parameters())

        if torch.cuda.is_available():
            self.device = "cuda:0"

        else:
            self.device = "cpu:0"

    def loss(self):
        pass

    def init_network(self):

        self.fc1 = torch.nn.Linear(self.in_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, self.out_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        out = x

        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out

    def optimize(self, x, y, batch_size, epochs):

        num_batch = x.shape[0] // batch_size
        remainder = x.shape[0] % batch_size


        # calculate number of steps based on epochs and batch size
        for i in range(num_batch):
            loss = self.loss(x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            loss.step()
