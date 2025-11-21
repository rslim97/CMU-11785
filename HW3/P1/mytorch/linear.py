import numpy as np

class Linear:

    def __init__(self, in_features, out_features):
        self.W = np.zeros(shape=(out_features, in_features))
        self.b = np.zeros(shape=(out_features, 1))
        self.N = 0
        self.x = None
        # self.dLdx = None
        self.dLdW = None
        self.dLdb = None
        self.ones = None
        self.update_weights = True

    def forward(self, x):
        # save for backprop
        self.N = x.shape[0]
        self.x = x
        self.ones = np.ones(shape=(self.N, 1))
        # print("x.shape ",x.shape)
        # print("W.shape ", self.W.shape)
        # print("b.shape ", self.b.shape)
        # print("self.ones ", self.ones.shape)
        z = x @ self.W.T + self.ones @ self.b.T
        return z

    def backward(self, dLdz):
        # print("dLdz linear backward:", dLdz)
        # print("dLdz.shape ", dLdz.shape)
        # print("self.W.shape ", self.W.shape)
        dLdx = dLdz @ self.W
        # print("dLdz.T.shape", dLdz.T.shape)
        # print("self.x.shape", self.x.shape)
        dLdW = dLdz.T @ self.x
        dLdb = dLdz.T @ self.ones
        # self.dLdx = dLdx
        self.dLdW = dLdW
        self.dLdb = dLdb
        print("")
        return dLdx

if __name__ == "__main__":
    lin = Linear(3, 2)
    x = np.random.randint(-5, 5, size=(10, 3))
    z = lin.forward(x)
    print(z)
    print(z.shape)
