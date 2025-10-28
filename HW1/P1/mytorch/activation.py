import numpy as np


class Activation:
    def __init__(self):
        self.x = None

    def forward(self, z):
        raise NotImplementedError()

    def backward(self, dLdx):
        raise NotImplementedError()


class Identity(Activation):
    def forward(self, z):
        self.x = z
        return self.x

    def backward(self, dLdx):
        dxdz = np.ones_like(self.x)
        dLdz = dLdx * dxdz
        return dLdz


class ReLU(Activation):
    def forward(self, z):
        self.x = np.maximum(z, np.zeros_like(z))
        return self.x

    def backward(self, dLdx):
        dxdz = np.zeros_like(self.x)
        dxdz[np.where(self.x > 0)] = 1
        dLdz = dLdx * dxdz
        return dLdz


class Tanh(Activation):
    def forward(self, z):
        self.x = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return self.x

    def backward(self, dLdx):
        dxdz = 1 - self.x**2
        dLdz = dLdx * dxdz
        return dLdz


class Sigmoid(Activation):
    def forward(self, z):
        self.x = 1 / (1 + np.exp(-z))
        return self.x

    def backward(self, dLdx):
        dLdz = dLdx * self.x * (1 - self.x)
        return dLdz


class Softmax(Activation):
    def forward(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        self.x = exp_z / sum_exp_z
        return self.x

    def backward(self, dLdx):
        N = self.x.shape[0]
        C = self.x.shape[1]
        dLdz = np.zeros(shape=(N, C))
        for k in range(N):
            J = np.zeros(shape=(C, C))
            for i in range(C):
                for j in range(C):
                    if i == j:
                        J[i, j] = self.x[k, i] * (1 - self.x[k, i])
                    else:
                        J[i, j] = -self.x[k, i] * self.x[k, j]
            dLdz[k, :] = dLdx[k, :] @ J
        return dLdz


if __name__ == "__main__":
    z = np.random.randint(-5, 5, (3, 4))
    relu = ReLU()
    x = relu.forward(z)
    dLdx = np.random.randint(0, 5, size=x.shape)
    dLdz = relu.backward(dLdx)
    print("z = ", z)
    print("x = ", x)
    print("dLdx = ", dLdx)
    print("dLdz = ", dLdz)
    print("")
    softmax = Softmax()
    x = softmax.forward(z)
    dLdZ = softmax.backward(dLdx)
    print("z = ", z)
    print("x = ", x)
    print("dLdx = ", dLdx)
    print("dLdz = ", dLdz)
