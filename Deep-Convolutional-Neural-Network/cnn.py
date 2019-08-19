import torch


class CNN(object):

    def __init__(self):

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


    def initialize_conv_weights(self):
        pass

    def initialize_dense_weights(self):
        pass

    def initialize_conv_bias(self):
        pass

    def initialize_dense_bias(self):
        pass
