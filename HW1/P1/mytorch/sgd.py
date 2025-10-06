import numpy as np
from .linear import Linear
from .batchnorm import BatchNorm1d

class SGD:

    def __init__(self, model, lr=0.001, momentum=0):
        # self.l = model.layers
        # self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = []
        self.v_b = []
        # exclude activation layers when updating weights
        self.layers = [layer for layer in model.layers if isinstance(layer, Linear) or isinstance(layer, BatchNorm1d)]
        # initialize weights for linear and batchnorm layers
        self.v_W = [
            np.zeros_like(layer.W) if isinstance(layer, Linear)
            else np.zeros_like(layer.gamma)
            for layer in self.layers
        ]
        self.v_b = [
            np.zeros_like(layer.b) if isinstance(layer, Linear)
            else np.zeros_like(layer.beta)
            for layer in self.layers
        ]

    def zero_grad(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.dLdW = np.zeros_like(layer.dLdW)
                layer.dLdb = np.zeros_like(layer.dLdb)
            elif isinstance(layer, BatchNorm1d):
                layer.dLdgamma = np.zeros_like(layer.dLdgamma)
                layer.dLdbeta = np.zeros_like(layer.dLdbeta)

    def step(self):
        for i, layer in enumerate(self.layers):
            if self.mu == 0:
                # without momentum
                if isinstance(layer, Linear):
                    layer.W = layer.W - self.lr * layer.dLdW
                    layer.b = layer.b - self.lr * layer.dLdb
                elif isinstance(layer, BatchNorm1d):
                    layer.gamma = layer.gamma - self.lr * layer.dLdgamma
                    layer.beta = layer.beta - self.lr * layer.dLdbeta
            else:
                # with momentum
                if isinstance(layer, Linear):
                    self.v_W[i] = self.mu * self.v_W[i] + layer.dLdW
                    self.v_b[i] = self.mu * self.v_b[i] + layer.dLdb
                    layer.W = layer.W - self.lr * self.v_W[i]
                    layer.b = layer.b - self.lr * self.v_b[i]
                elif isinstance(layer, BatchNorm1d):
                    self.v_W[i] = self.mu * self.v_W[i] + layer.dLdgamma
                    self.v_b[i] = self.mu * self.v_b[i] + layer.dLdbeta
                    layer.gamma = layer.gamma - self.lr * self.v_W[i]
                    layer.beta = layer.beta - self.lr * self.v_b[i]
