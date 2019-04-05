import numpy as np
import torch
from tqdm import tqdm
import time


# torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)


class CNN(object):
    # Make prediction loop
    # sort out pooling
    # make training loop
    # make backpropogation method
    # add bias updates

    def __init__(self, num_convs, num_dense, output_shape=1, input_shape=[28, 28, 1]):

        self.historical_cost = []


        # define network topology
        self.num_convs = num_convs
        self.num_dense = num_dense
        self.output_shape = output_shape
        self.input_shape = input_shape

        # define weights
        self.dense_w = {}
        self.conv_w = {}
        self.dense_b = {}
        self.conv_b = {}
        self.pool_a = {}
        self.conv_a = {}
        self.dense_a = {}
        self.conv_z = {}
        self.dense_z = {}

        # initiate weights
        self.initiate_weights()

    # initiates dense and conv weights and bias
    def initiate_weights(self):

        # initiating dense weights
        for i in range(1, self.num_dense):

            self.dense_w["w" + str(i)] = torch.rand(self.output_shape + 1, self.output_shape + 1)

        self.dense_w["w" + str(self.num_dense)] = torch.rand(self.output_shape, self.output_shape + 1)

        # initiate convolutional weights

        self.conv_w["w1"] = torch.rand([2, 2, 1, 1])
        self.conv_w["w2"] = torch.rand([2, 2, 1, 1])
        self.conv_w["w3"] = torch.rand([2, 2, 1, 1])
        self.conv_w["w4"] = torch.rand([2, 2, 1, 1])

    # make a prediction based on x
    def predict(self, x):

        self.conv_a["a0"] = x

        for i in range(self.num_convs+1):
            self.conv_forward(self.conv_a["a" + str(i)], conv_layer=i+1, step=2)
            self.conv_a["a" + str(i+1)] = self.tanh(self.conv_z["z" + str(i+1)])
            print("Calculated Convolution")


        self.dense_forward()
        print("Calculated Dense")

        

    # traing conv net model
    def train(self, x, y, iterations=1, alpha=0.1):

        print("    Training Convolutional Neural Network")
        for i in tqdm(range(iterations)):

            self.predict(x)
            cost = self.mean_square_error(y)
            self.historical_cost.append(cost)
            self.calc_dense_updates(cost)
            self.dense_backprop()

    # single convolution operation
    def single_conv(self, x, conv_layer=1):

        w = "w" + str(conv_layer)
        b = "b" + str(conv_layer)
        n = (torch.tensor(x).type(torch.DoubleTensor) * self.conv_w[w].type(torch.DoubleTensor))
        return torch.sum(n).type(torch.DoubleTensor)

    # convolutional forward
    def conv_forward(self, x, conv_layer=1, step=1):

        # height, width, thickness
        w = "w" + str(conv_layer)
        x_h, x_w, x_t = x.shape[0], x.shape[1], x.shape[2]
        w_h, w_w, w_t, w_f = self.conv_w[w].shape[0], self.conv_w[w].shape[1], self.conv_w[w].shape[2], \
                             self.conv_w[w].shape[3]

        h_num = self.calc_num_steps(step, x_h, w_h)
        w_num = self.calc_num_steps(step, x_w, w_w)
        t_num = self.calc_num_steps(step, x_t, w_t)

        # print(h_num, w_num, t_num, w_f)
        # print(w_h, w_w, w_t, w_f)

        self.conv_z["z" + str(conv_layer)] = torch.rand(h_num, w_num, t_num, w_f)

        for h in range(h_num):
            for w in range(w_num):
                for t in range(t_num):
                    for f in range(w_f):
                        x_slice = x.narrow(0, h * step, w_h).narrow(1, w * step, w_w).narrow(2, t * step, w_t)
                        self.conv_z["z" + str(conv_layer)][h, w, t, f] = self.single_conv(x_slice,
                                                                                          conv_layer=conv_layer)
        self.conv_z["z" + str(conv_layer)] = self.pad(self.conv_z["z" + str(conv_layer)], pad_num=1)

    # single pool operation
    @staticmethod
    def single_pool(x, pool_type="average"):

        if pool_type == "average":
            return torch.sum(x) / x.numel()

        if pool_type == "max":
            return x.max()

    # pooling forward
    def pool_forward(self, x, pool_layer=1, output_shape=[1, 1, 1], step=1, pool_type="average"):

        # height, width, thickness
        x_h, x_w, x_t, x_f = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        h_num, w_num, t_num = output_shape[0], output_shape[1], output_shape[2]

        w_h = self.calc_window_side(x_h, h_num)
        w_w = self.calc_window_side(x_w, w_num)
        w_t = self.calc_window_side(x_t, t_num)

        self.pool_a["a" + str(pool_layer)] = torch.zeros(h_num, w_num, t_num, x_f)

        for h in range(h_num):
            for w in range(w_num):
                for t in range(t_num):
                    for f in range(x_f):
                        x_slice = x.narrow(0, h * step, w_h).narrow(1, w * step, w_w).narrow(2, t * step, w_t)
                        self.pool_a["a" + str(pool_layer)][h, w, t, f] = self.single_pool(x, pool_type)

    # pad torch tensor with number a
    def pad(self, a, pad_num=1):

        padding = [pad_num, pad_num]

        if len(a.shape) == 4:
            padding = [0, 0, 0, 0, pad_num, pad_num, pad_num, pad_num]

        if len(a.shape) == 3:
            padding = [0, 0, pad_num, pad_num, pad_num, pad_num]

        if len(a.shape) == 2:
            padding = [pad_num, pad_num, pad_num, pad_num]

        
        return torch.nn.functional.pad(a, pad=padding)

    # calculate the number of conv stepts from the perameters
    def calc_num_steps(self, step, x, w):
        return int((x - w) / (step)) + 1

    # calculate the length of window from perameters
    def calc_window_side(self, step, x, length):
        return int(x - ((length - 1) * step))

    # making dense prediction from convolutional layer
    def dense_forward(self):

        self.dense_a["a0"] = self.conv_a["a" + str(self.num_convs)].reshape(self.conv_a["a" + str(self.num_convs)].numel())

       


        self.dense_w["w1"] = torch.rand(self.dense_a["a0"].shape[0] , self.output_shape + 1).t()

        for i in range(self.num_dense):


            if i != self.num_dense-2:
                self.dense_z["z" + str(i+1)] = torch.matmul(self.dense_a["a"+str(i)], self.dense_w["w"+str(i+1)].t()) #+ self.dense_b["b" + str(i+1)]
                self.dense_a["a"+str(i+1)] = self.tanh(self.dense_z["z" + str(i+1)])

            else:
                self.dense_z["z" + str(i + 1)] = torch.matmul(self.dense_a["a" + str(i)], self.dense_w["w" + str(i + 1)].t()) #+ self.dense_b["b" + str(i + 1)]
                self.dense_a["a" + str(i + 1)] = self.tanh(self.dense_z["z" + str(i + 1)])


    # sigmoid activation function
    def sigmoid(self, z):
        return torch.sigmoid(z)

    # tanh activation funciton
    def tanh(self, z):
        return torch.tanh(z)

    # derivative of sigmoid activation function
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # derivative of tanh activation 
    def tanh_prime(self, z):
        return 1 - self.tanh(z) ** 2

    # updating dense weights
    def dense_backprop(self):
        
        for i in range(self.num_dense):
            self.w = self.w - self.alpha*self.dense_w_update["w" + str(i)]

    # updates for dense weights
    def calc_dense_updates(self, cost):

        # updates the first set of wait updates to get it started
        self.dense_w_update['w' + str(self.num_dense)] = torch.matmul(self.dense_a["a" + str(self.num_dense)]*self.cost - self.mean_square_error(y))

        # loop through all of the dense layers in reverse generating updates
        for i in reversed(range(self.num_dense)):
            self.dense_w_update["w" + str(i)] = torch.matmul(self.dense_w_update["w" + str(i+1)])

    # mean squared error cost function
    def mean_square_error(self, y):
        return 0.5*torch.pow((self.dense_a["a"+str(self.num_dense)] - y), 2)

    # derivative of mean squared error cost
    def mean_square_prime(self, y):
        return self.dense_a["a" + str(self.num_dense)] - y

    # log liklihood cost function
    def log_liklihood(self, y):
        h = self.dense_a["a" + str(self.num_dense)]
        return y*torch.log(h) + (1-y)*torch.log(1-h)

    # derivative of log liklihood cost funciton
    def log_liklihood_prime(self, y):
        h = self.dense_a["a" + str(self.num_dense)]
        return y/h + (1-y)/(1-h)


