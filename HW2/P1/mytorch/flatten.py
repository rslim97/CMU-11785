import numpy as np

class Flatten:
    def forward(self, x):
        self.x = x
        # keep the batch_size dimension and flatten other dims
        z = x.reshape(x.shape[0], -1)
        return z

    def backward(self, dLdz):
        dLdz = np.reshape(dLdz, self.x.shape)
        return dLdz