import torch as tf

class CNN(object):

    # TO DO:
    # - initalize dense weights
    # - initialize conv weights
    # - initalize pool weights lol :)
    # - single conv
    # - single pool
    # - conv forward
    # - dense forward
    # - pool forward
    # - conv backward
    # - dense backward

    def __init__(self):
        
        # initalize weights
        self.dense_w = {}
        self.conv_w = {}

        # initialize z
        self.dense_z = {}
        self.conv_z = {}
        self.pool_z = {}

        # initialize activations
        self.dense_a = {}
        self.conv_a = {}
        self.pool_a = {}

    def initialize_dense_weights(self):
        pass

    def initialize_conv_weights(self):
        pass

    def single_conv(self):
        pass

    def single_avg_pool(self):
        pass

    def conv_forward(self):
        pass

    def pool_forward(self):
        pass

    def dense_forward(self):
        pass

    def conv_backward(self):
        pass

    def dense_backward(self):
        pass