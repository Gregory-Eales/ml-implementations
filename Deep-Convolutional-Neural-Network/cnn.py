import torch

class CNN(object):

    def __init__(self, input_shape=10, output_size=1, num_conv=5, num_dense=1):

        # initialize network parameters
        self.inpute_shape = input_shape
        self.output_size = output_size
        self.num_conv = num_conv + 1
        self.num_dense = num_dense + 1
        self.cuda = None

        # initialize weights and bias
        self.conv_w = None
        self.dense_w = None
        self.conv_b = None
        self.dense_b = None

        # populate weights and bias
        self.initialize_conv_weights()
        self.initialize_conv_bias()
        self.initialize_dense_weights()
        self.initialize_dense_bias()

    def initialize_cuda(self):

        if torch.cuda.is_available():
            self.cuda = True

        else:
            self.cuda = False


    def initialize_conv_weights(self):

        # init dict
        self.conv_w = {}

        # loop through layers and initialize network weights
        for i in range(1, self.num_layers):
            self.conv_w["w"+str(i)] = torch.randn(self.conv_w, self.conv_w)

    def initialize_dense_weights(self):

        # init dict
        self.dense_w = {}

        # loop thorugh layers and initialize dense
        for i in range(1, self.num_dense):
            self.dense_w["w" + str(i)] = torch.rand(self.output_size+(self.num_dense-(i-1)))

    def initialize_conv_bias(self):
        pass

    def initialize_dense_bias(self):
        pass

    def single_conv(self):
        pass

    def single_pool(self):
        pass
