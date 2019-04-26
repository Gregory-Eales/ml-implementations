import torch

class CNN(object):

    # TO DO:
    # - conv forward
    # - dense forward
    # - pool forward
    # - conv backward
    # - dense backward

    def __init__(self, dense_length, conv_length, x_shape, y_shape):

        # conv_length = the number of convolutional layers
        # dense_length = the number of dense layers
        # x_shape = the shape of the input data
        # y_shape = the shape of the output data
        
        # init params
        self.dense_length = dense_length
        self.conv_length = conv_length
        self.x_shape = x_shape
        self.y_shape = y_shape

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

        # run intialization functions
        self.initialize_weights()


    def initialize_dense_weights(self):
        
        # need output shape of last conv layer
        output = self.conv_w["w" + str(self.conv_length)].numel()

        self.dense_w["w1"] = torch.rand(output, y.shape[1] + 1)

        for i in range(2, self.dense_length+1):
            
            if i != self.dense_length:
                self.dense["w"+str(i)] = torch.rand(y.shape[1]+1, y.shape[1]+1)

            else:
                self.dense["w"+str(i)] = torch.rand(y_shape[1]+1, y_shape[1])

    def initialize_conv_weights(self):
        
        for i in range(self.conv_length):
            self.conv_w["w" + str(i+1)] = torch.rand(64/(2**i), 3, 3)

    def initialize_weights(self):
        self.initialize_conv_weights()
        self.initialize_dense_weights()

    def single_conv(self, x, w, activation="tanh", conv_layer=1):
        
        if activation == "tanh":
            self.conv_z["z"+str(conv_layer)] = torch.sum(x*w)
            return torch.tanh(self.conv_z["z"+str(conv_layer)])

        if activation == "relu":
            self.conv_z["z"+str(conv_layer)] = torch.sum(x*w)
            return torch.nn.functional.relu(self.conv_z["z"+str(conv_layer)])

        if activation == "sigmoid":
            self.conv_z["z"+str(conv_layer)] = torch.sum(x*w)
            return torch.sigmoid(self.conv_z["z"+str(conv_layer)])

    def single_avg_pool(self, z, kernal_size=3, activation="tanh"):

        if activation == "tanh":
            return torch.tanh(torch.sum(z)/kernal_size**2)

        if activation == "relu":
            return torch.nn.functional.relu(torch.sum(z)/kernal_size**2)

        if activation == "sigmoid":
            return torch.sigmoid(torch.sum(z)/kernal_size**2)
        
    def conv_forward(self, conv_layer=1):
        
        # get number of steps in each direction

        # loop through each step and calculate conv a
        x, y, z = 1, 1, 1
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    for l in range(w):
                        pass

    def pool_forward(self, pool_layer=1):
        pass

    def dense_forward(self):
        
        # flatten last conv activation 
        self.dense_a["a0"] = self.conv_a["a"+str(self.conv_length)].view(-1, 1)

        for i in range(self.dense_length):

            if i+1 != self.dense_length:
                self.dense_z["z"+str(i+1)] = torch.matself.dense_w["w"+str(i+1)]
                self.dense_a["a"+str(i+1)] = torch.tanh(self.dense_z["z"+str(i+1)])

            else:
                self.dense_z["z"+str(i+1)] = torch.matself.dense_w["w"+str(i+1)]
                self.dense_a["a"+str(i+1)] = torch.sigmoid(self.dense_z["z"+str(i+1)])

    def conv_backward(self):
        pass

    def dense_backward(self):
        pass

    def sigmoid_prime(self):
        pass

    def tanh_prime(self):
        pass

    def relu_prime(self):
        pass