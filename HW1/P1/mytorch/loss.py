import numpy as np

class Loss:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

class MSELoss(Loss):
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

class CrossEntropyLoss(Loss):
    def forward(self, x, y):
        self.y = y
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        self.x = exp_x / sum_exp_x
        N = x.shape[0]
        cross_entropy = np.sum(- y * np.log(self.x), axis=1)
        sum_cross_entropy = np.sum(cross_entropy, axis=0)
        mean_cross_entropy = sum_cross_entropy / N
        return mean_cross_entropy

    def backward(self):
        N = self.x.shape[0]
        dLdx = (self.x - self.y) / N
        return dLdx

if __name__ == "__main__":
    x = np.random.randint(0,5, size=(2,5))
    y = np.random.randint(0,5, size=(2,5))
    mseloss = MSELoss()
    L = mseloss.forward(x, y)
    dLdxN = mseloss.backward()

    print("x", x)
    print("y", y)
    print("L ", L)
    print("dLdxN", dLdxN)
    print("")

    x = np.zeros(shape=(2,5))
    y = np.zeros_like(x)
    y[0, 2] = 1
    y[1, 1] = 1

    crossentropy = CrossEntropyLoss()
    L = crossentropy.forward(x, y)
    dLdxN = crossentropy.backward()

    print("x", x)
    print("y", y)
    print("L ", L)
    print("dLdxN", dLdxN)
