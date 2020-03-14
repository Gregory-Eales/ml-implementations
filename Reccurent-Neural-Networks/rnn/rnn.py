import torch

class RNN(torch.nn.Module):

    def __init__(self, sequence_size=5, alpha=0.01):

        super(RNN, self).__init__()

        self.sequence_size = sequence_size
        self.layers = {}
        self.define_network()

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=alpha)

        # get device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        print(torch.cuda.is_available())
        print(self.device)
        self.to(self.device)


    def define_network(self):

        self.fc1 = torch.nn.Linear(1, 1)
        self.fc2 = torch.nn.Linear(2, 1)
        self.fc3 = torch.nn.Linear(2, 1)
        self.fc4 = torch.nn.Linear(2, 1)
        self.fc5 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.leaky_relu = torch.nn.LeakyReLU()


    def forward(self, x):

        out = self.fc1(x[:,0:1])
        self.sigmoid(out)
        out = torch.cat([out, x[:,1:2]], dim=1)


        out = self.fc2(out)
        self.sigmoid(out)
        out = torch.cat([out, x[:,2:3]], dim=1)

        out = self.fc3(out)
        self.sigmoid(out)
        out = torch.cat([out, x[:,3:4]], dim=1)

        out = self.fc4(out)
        self.sigmoid(out)
        out = torch.cat([out, x[:,4:5]], dim=1)

        out = self.fc5(out)
        self.sigmoid(out)

        return out

    def optimize(self, x, y, iter):

        for i in range(iter):
            prediction = self.forward(x)
            loss = self.loss(prediction, y)
            loss.backward()
            self.optimizer.step()


def main():
    rnn = RNN(sequence_size=5)
    x = torch.rand(100, 5)
    y_prime = rnn.forward(x)


if __name__ == "__main__":
    main()
