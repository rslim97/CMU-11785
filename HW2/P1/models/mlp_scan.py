# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from HW2.P1.mytorch.flatten import *
from HW2.P1.mytorch.Conv1d import *
from HW2.P1.mytorch.linear import *
from HW2.P1.mytorch.activation import *
from HW2.P1.mytorch.loss import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        self.conv1 = Conv1d(24, 8, 8, 4)
        self.conv2 = Conv1d(8, 16, 1, 1)
        self.conv3 = Conv1d(16, 4, 1, 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights
        # Load them appropriately into the CNN -
        #   1. For each conv layer, have a look at the shape of its weight matrix
        #   2. Look at the shapes of w1, w2 and w3
        #   3. Figure out appropriate reshape and transpose operations

        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = w1.T.reshape((8, 8, 24)).transpose(0, 2, 1)  # originally (192,8)
        self.conv2.conv1d_stride1.W = w2.T.reshape((16, 1, 8)).transpose(0, 2, 1)  # originally (8, 16)
        self.conv3.conv1d_stride1.W = w3.T.reshape((4, 1, 16)).transpose(0, 2, 1)  # originally (16, 4)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
            z (np.array): (batch_size, out_channel, out_width)
        """
        l = len(self.layers)
        z = x
        for i in range(l):
            z = self.layers[i].forward(z)
        return z

    def backward(self, dLdz):
        l = len(self.layers)
        dLdx = dLdz
        for i in reversed(range(l)):
            dLdx = self.layers[i].backward(dLdx)
        return dLdx

class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1d(24, 2, 2, 2)
        self.conv2 = Conv1d(2, 8, 2, 2)
        self.conv3 = Conv1d(8, 4, 2, 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = w1[:48, :2].T.reshape((2, 2, 24)).transpose(0, 2, 1)
        self.conv2.conv1d_stride1.W = w2[:4, :8].T.reshape((8, 2, 2)).transpose(0, 2, 1)
        self.conv3.conv1d_stride1.W = w3[:, :].T.reshape((4, 2, 8)).transpose(0, 2, 1)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
            z (np.array): (batch_size, out_channel, out_width)
        """
        l = len(self.layers)
        z = x
        for i in range(l):
            z = self.layers[i].forward(z)
        return z

    def backward(self, dLdz):
        l = len(self.layers)
        dLdx = dLdz
        for i in reversed(range(l)):
            dLdx = self.layers[i].backward(dLdx)
        return dLdx