import torch
import tqdm

class CNN(object):

    def __init__(self, input_shape=10, output_size=1, num_conv=5, num_dense=1):

        # initialize network parameters
        self.input_shape = input_shape
        self.output_size = output_size
        self.num_conv = num_conv + 1
        self.num_dense = num_dense + 1
        self.cuda = None

        # initialize weights and bias
        self.conv_w = None
        self.dense_w = None
        self.conv_b = None
        self.dense_b = None

        # check if cuda is available
        self.initialize_cuda_state()

        # populate weights and bias
        self.initialize_conv_weights()
        self.initialize_conv_bias()
        self.initialize_dense_weights()
        self.initialize_dense_bias()


    ##########################
    # Initialization Methods #
    ##########################


    def initialize_cuda_state(self):

        if torch.cuda.is_available():
            self.cuda = True

        else:
            self.cuda = False


    def initialize_conv_weights(self):

        # init dict
        self.conv_w = {}

        # loop through layers and initialize network weights
        for i in range(1, self.num_conv):
            self.conv_w["w"+str(i)] = torch.randn(self.input_shape, self.input_shape)

    def initialize_dense_weights(self):

        # init dict
        self.dense_w = {}

        # loop thorugh layers and initialize dense
        for i in range(1, self.num_dense):
            self.dense_w["w" + str(i)] = torch.rand(self.output_size+(self.num_dense-(i-1)))

    def initialize_conv_bias(self):

        # init dict
        self.conv_b = {}

        # loop through layers and initialize conv bias
        for i in range(1, self.num_conv):
            self.conv_b["b"+str(i)] = 0

    def initialize_dense_bias(self):

        # init dict
        self.dense_b = {}

        # loop through layers and initialize conv bias
        for i in range(1, self.num_dense):
            self.dense_b["b"+str(i)] = 0

    ######################
    # Activation Methods #
    ######################

    def sigmoid_prime(self, z):
        sig = torch.sigmoid(z)
        return sig*(1-sig)

    def tanh_prime(self):
        tan = torch.tanh(z)
        return 1 - (tan)**2

    ###############################
    # Forward Propogation Methods #
    ###############################

    def pad(self, z, pad, value):
        return torch.nn.functional.pad(z, [pad, pad, pad, pad], mode='constant', value=0)

    def single_conv(self, z, b, activation):
        return torch.sum(activation(z + b), dim=[1, 2]) / z.shape[0]**2

    def single_max_pool(self):
        pass

    def single_avg_pool(self, z):
        return torch.sum(z, dim=[1, 2]) / z.shape[0]**2

    def conv_forward(self, num_layer):

        # get input dimensions
        width, height = self.conv_a["a" + str(num_layer)].shape[2], self.conv_a["a" + str(num_layer)].shape[1]

        # get weight dimensions
        w_w, h_w = self.w["w" + str()]

        # calculate the number of convolutional steps needed to take

        for x_l in range(width):
            for y_l in range(height):
                pass


    def pool_forward(self):
        pass

    def predict(self, x):
        pass

    ################################
    # Backward Propogation Methods #
    ################################

    def cost(self):
        pass

    def cost_prime(self):
        pass

    def calc_conv_updates(self):
        pass

    def calc_dense_updates(self):
        pass

    def calc_updates(self):
        pass

    def update_weights(self):
        pass

    def train(self):
        pass

def main():
    x = torch.randn(10, 80, 80)
    y = torch.randn(10, 1)
    cnn = CNN()
    a = cnn.single_conv(x, 10, torch.sigmoid)
    print(a.shape)


if __name__ == "__main__":
    main()
