from HW2.P1.mytorch.flatten import *
from HW2.P1.mytorch.Conv1d import *
from HW2.P1.mytorch.linear import *
from HW2.P1.mytorch.activation import *
from HW2.P1.mytorch.loss import *


class CNN:
    """
    A simple convolutional neural network
    """
    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        :param input_width          : int: The width of the input to the first convolutional layer
        :param num_input_channels   : int : Number of channels for the input layer
        :param num_channels         : [int] : List containing number of (output) channels for each conv
        :param kernel_sizes         : [int] : List containing kernel width for each conv layer
        :param strides:             : [int] : List containing stride size fo each conv layer
        :param num_linear_neurons   : int : Number of neurons in the linear layer
        :param activations          : [obj] : List of objects corresponding to the activation fn for each conv layer
        :param conv_weight_init_fn  : fn : Function to init each conv layers weights
        :param bias_init_fn         : fn : Function to initialize each conv layers AND the linear layers bias to 0
        :param linear_weight_init_fn: fn :Function to iinitialize the linear layers weights
        :param criterion            : obj : Object to the criterion (SoftMaxCrossEntropy) to be used
        :param lr                   : float : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # self.conv1 = Conv1d(num_input_channels, num_channels[0], kernel_sizes[0], strides[0])
        # self.conv2 = Conv1d(num_channels[0], num_channels[1], kernel_sizes[1], strides[1])
        # self.conv3 = Conv1d(num_channels[1], num_channels[2], kernel_sizes[2], strides[2])
        # self.convolutional_layers = [self.conv1, self.conv2, self.conv3]
        self.convolutional_layers = []
        for i in range(self.nlayers):
            self.convolutional_layers.append(
                Conv1d(num_input_channels,
                       num_channels[i],
                       kernel_sizes[i],
                       strides[i],
                       weight_init_fn = conv_weight_init_fn,
                       bias_init_fn = bias_init_fn)
            )
            num_input_channels = num_channels[i]

        self.flatten = Flatten()
        out_width = input_width
        for i in range(self.nlayers):
            out_width = (out_width - kernel_sizes[i]) //  strides[i] + 1

        in_features = out_width * num_channels[-1]
        self.linear_layer = Linear(in_features, num_linear_neurons)

        # Don't change this -->
        out_features, in_features = self.linear_layer.W.shape
        if linear_weight_init_fn is not None:
            self.linear_layer.W = linear_weight_init_fn(out_features, in_features)
        if bias_init_fn is not None:
            self.linear_layer.b = bias_init_fn(out_features)
        # <---------------------

    def forward(self, x):
        """
        Argument:
            x (np.array) : (batch_size, num_input_channels, input_width)
        Return :
            z (np.array) : (batch_size, num_linear_neurons)
        """

        # Iterate through each layer
        # <------------------
        # save output (necassary for error and loss)
        for i in range(self.nlayers):
            x = self.convolutional_layers[i].forward(x)
            x = self.activations[i].forward(x)
        x = self.linear_layer.forward(self.flatten.forward(x))
        self.z = x
        return self.z

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch_size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion.forward(self.z, labels).sum()
        grad = self.criterion.backward()

        # Iterate through each layer in reverse order
        # <----------------
        # linear backward and flatten backward
        grad = self.flatten.backward(self.linear_layer.backward(grad))
        for i in reversed(range(self.nlayers)):
            grad = self.activations[i].backward(grad)
            grad = self.convolutional_layers[i].backward(grad)
        return grad

    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW = np.zeros(self.linear_layer.W.shape)
        self.linear_layer.dLdb = np.zeros(self.linear_layer.b.shape)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (self.convolutional_layers[i].conv1d_stride1.W -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW)
            self.convolutional_layers[i].conv1d_stride1.b = (self.convolutional_layers[i].conv1d_stride1.b -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb)

        self.linear_layer.W = self.linear_layer.W - self.lr * self.linear_layer.dLdW
        self.linear_layer.b = self.linear_layer.b - self.lr * self.linear_layer.dLdb

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        self.train_mode = False