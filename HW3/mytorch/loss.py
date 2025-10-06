import numpy as np

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class MSELoss(Criterion):
    def forward(self, x, y):
        self.x = x
        self.y = y
        N = x.shape[0]
        C = x.shape[1]
        squared_error = (x - y) * (x - y)
        sum_of_squared_error = np.sum(squared_error, axis=(0, 1))
        mean_squared_error_loss = sum_of_squared_error / (N * C)
        # print("mean_squared_error_loss ", mean_squared_error_loss)
        return mean_squared_error_loss

    def backward(self):
        N = self.x.shape[0]
        C = self.x.shape[1]
        dLdx = 2 * (self.x - self.y) / (N * C)
        # print("dLdx", dLdx)
        return dLdx

class SoftmaxCrossEntropy(Criterion):
    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.y = y
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        self.x = exp_x / sum_exp_x
        # N = x.shape[0]
        cross_entropy = np.sum(- y * np.log(self.x), axis=1)
        sum_cross_entropy = np.sum(cross_entropy, axis=0)
        # mean_cross_entropy = sum_cross_entropy / N
        mean_cross_entropy = sum_cross_entropy
        return mean_cross_entropy

    def backward(self):
        # N = self.x.shape[0]
        # dLdx = (self.x - self.y) / N
        dLdx = self.x - self.y
        return dLdx

