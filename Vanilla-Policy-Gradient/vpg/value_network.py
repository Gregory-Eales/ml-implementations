import torch

class ValueNetwork(torch.nn.Module):

    def __init__(self, alpha, input_dims, output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims

        # inherit from nn module class
        super(ValueNetwork, self).__init__()

        # initialize_network
        self.initialize_network()

        # define optimizer
        self.optimizer = torch.optim.SGD(lr=alpha, params=self.parameters())

        # define loss
        self.loss = torch.nn.MSELoss()


    # initialize network
    def initialize_network(self):

		# define network components
        self.batch_norm1 = torch.nn.BatchNorm1d(self.input_dims)
        self.fc1 = torch.nn.Linear(self.input_dims, 3)
        self.fc2 = torch.nn.Linear(3, self.output_dims)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def predict(self, x):
        out = self.batch_norm1(x)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu1(out)
        return out

    def update(self, x, y, iter):

        for i in range(iter):

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # make prediction
            prediction = self.predict(x)

            # calculate loss
            loss = self.loss(prediction, y)
            print("Iteration: {}".format(i))
            print("Loss: {}".format(loss))

            # optimize
            loss.backward()
            self.optimizer.step()


def main():

    x = torch.ones(100, 2)
    y = torch.ones(100, 2)

    # defin value ValueNetwork
    vn = ValueNetwork(alpha=0.01, input_dims=2, output_dims=2)

    vn.update(x, y, 100)

if __name__ == "__main__":
    main()
