import torch

class RNN(object):

    def __init__(self, x_shape, y_shape):
        
        # store init variables
        self.x_shape = x_shape
        self.y_shape = y_shape

        # init dense params
        self.dense_w = {}
        self.dense_z = {}
        self.dense_b = {}
        self.dense_a = {}

        # init reccurent params
        self.rec_w = {}
        self.rec_z = {}
        self.rec_b = {}
        self.rec_a = {}

    def initialize_weights(self):
        self.init_reccurent_weights()
        self.init_dense_weights()

    def init_reccurent_weights(self):
        pass

    def init_dense_weights(self):
        pass

    def single_reccurent(self):
        pass

    def reccurent_forward(self):
        pass

    def dense_forward(self):
        pass

    def reccurent_backward(self):
        pass

    def sigmoid_prime(self, z):
        a = torch.sigmoid(z)
        return a*(a-1)

    def tanh_prime(self, z):
        return 1 - (torch.tanh(z)**2)

    def relu_prime(self, z):
        z[z>0] = 1
        z[z<0] = 0
        return z






