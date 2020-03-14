import torch

class RNN(torch.nn.Module):

    def __init__(self, sequence_size):

        super(RNN, self).__init__()

        self.sequence_size = sequence_size
        self.layers = {}

        self.leaky_relu = torch.nn.LeakyReLU()
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=0.1)

    def define_network(self):

        self.layers["l1"] = torch.nn.Linear(1, 1)

        for i in range(1, self.sequence_size):
            self.layers["l"+str(i+1)] = torch.nn.Linear(2, 1)

    def forward(self, x):


        out = self.layers["l1"](x[0])

        for i in range(1, self.sequence_size):
            input = torch.cat([out, x[i]]).reshape(1, 2)
            out = self.layers["l"+str(i+1)](input)

        return out

    def update(self):
        pass


def main():
    rnn = RNN(sequence_size=5)
    x = torch.rand(5, 1)


if __name__ == "__main__":
    main()
