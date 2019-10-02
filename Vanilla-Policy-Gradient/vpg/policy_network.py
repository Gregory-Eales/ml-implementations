import torch

class PolicyNetwork(torch.nn.Module):

	def __init__(self, alpha, input_dims, output_dims):

		# inherit from nn module class
		super(PolicyNetwork, self)__init__()

		# initialize network
		self.initialize_network()

        # define optimizer
        self.optimizer = torch.optim.SGD(lr=alpha)

        # define loss
        self.loss = torch.nn.CrossEntropyLoss()

	# initialize network
	def initialize_network(self):

		# define network components
		self.batch_norm1 = torch.nn.BatchNorm1d()
		self.fc1 = torch.nn.Linear(4)
		self.relu1 = torch.nn.ReLU()

	def predict(x):
		out = self.batch_norm(x)
		out = self.fc1(out)

	# training loop
	def update(self, x, y, iter):

        for i in range(iter):

            # zero the parameter gradients
            optimizer.zero_grad()

            # make prediction
            prediction = self.predict(x)

            # calculate loss
            loss = self.loss(prediction, y)

            # optimize
            loss.backward()
            self.optimizer.step()



def main():
    pass

if __name__ == "__main__":
    main()
