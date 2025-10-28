import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        self.debug = debug
        self.W = np.zeros(shape=(out_features, in_features))
        self.b = np.zeros(shape=(out_features, 1))
        self.N = 0
        self.x = None
        self.dLdx = None
        self.dLdW = None
        self.dLdb = None
        self.ones = None

    def forward(self, x):
        # save for backprop
        self.N = x.shape[0]
        self.x = x
        self.ones = np.ones(shape=(self.N, 1))
        z = x @ self.W.T + self.ones @ self.b.T
        return z

    def backward(self, dLdz):
        dLdx = dLdz @ self.W
        dLdW = dLdz.T @ self.x
        dLdb = dLdz.T @ self.ones
        self.dLdW = dLdW
        self.dLdb = dLdb
        if self.debug:
            self.dLdA = dLdx
        return dLdx


if __name__ == "__main__":
    lin = Linear(3, 2)
    x = np.random.randint(-5, 5, size=(10, 3))
    z = lin.forward(x)
    print(z)
    print(z.shape)
