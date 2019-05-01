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
        #self.initialize_weights()


    def predict(self, x):

        self.conv_a["a0"] = x
        for i in range(self.conv_length):
            self.conv_forward(conv_layer=i)

        self.dense_forward()

        return self.denes_a["a" + str(self.dense_length)]

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

            if i == 0:
                self.conv_w["w" + str(i+1)] = torch.rand(3, 3, 1, 64/(2**i))

            else:
                self.conv_w["w" + str(i+1)] = torch.rand(3, 3, 32/(2**i-1), 64/(2**i))


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

        x_shape = self.conv_a["a" + str(conv_layer-1)]
        w_shape = self.conv_w["w" + str(conv_layer)]
        l, w, h, f = self.get_wind_shape(x_shape, w_shape)
      
        
        # get number of steps in each direction

        # loop through each step and calculate conv a

        self.conv_z["z" + str(conv_layer)] = torch.zeros(l, w, f)
        
        for j in range(l):
            for i in range(w):
                for k in range(f):

                    y = self.conv_w["w"+str(conv_layer)].shape[0]
                    x = self.conv_w["w"+str(conv_layer)].shape[1]
                    z = self.conv_w["w"+str(conv_layer)].shape[2]

                    conv_a_slice = self.conv_a["a"+str(conv_layer-1)][j*y:j*(y+1)][i*x:i*(x+1)][k:]
                    self.conv_z["z"+str(conv_layer)][j][i][0][k] =  self.single_conv(conv_a_slice, self.conv_w["w"+str(conv_layer)], conv_layer=conv_layer)
                    

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

    def sigmoid_prime(self, ):
        a = torch.sigmoid(z)
        return a*(1-a)

    def tanh_prime(self, z):
        return 1 - torch.tanh(z)**2

    def relu_prime(self, z):
         z[z>0] = 1
         z[z<0] = 0
         return z

    def get_wind_shape(self, x_shape, w_shape, step=1):

        x_l, x_w, x_h = x_shape[0], x_shape[1], x_shape[2]
        w_l, w_w, w_h, w_f = w_shape[0], w_shape[1], w_shape[2], w_shape[3]

        l = (x_l - w_l)/step + 1
        w = (x_w - w_w)/step + 1
        
        # make sure the weights and next conv have same filter dimension
        assert w_h == x_h
        
        return l, w, x_h, w_f

    def get_pool_wind_shape(self, x_shape, kernal_size=3, step=1):

        x_l, x_w, x_h = x_shape[0], x_shape[1], x_shape[2]
        p_l, p_w, p_h = kernal_size, kernal_size, kernal_size
        l = (x_l - p_l)/step + 1
        w = (x_w - p_w)/step + 1
        h = (x_h - p_h)/step + 1
        
        return l, w, h






