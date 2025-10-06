import numpy as np
from .resampling import *

def f1(C_out, C_in, K1, K2):
    return np.random.randint(0, 5, size=(C_out, C_in, K1, K2))

def f2(C_out):
    return np.random.randint(0, 3, size=(C_out, 1))

class Conv2d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn):
        self.C_in = in_channels
        self.C_out = out_channels
        self.K = kernel_size

        self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = bias_init_fn(out_channels)

        self.dLdW = None
        self.dLdb = None

    def forward(self, x):
        self.x = x
        N, C_in, H_in, W_in = x.shape
        H_out = H_in - self.K + 1
        W_out = W_in - self.K + 1
        z = np.empty(shape=(N, self.C_out, H_out, W_out), dtype=np.float64)
        for i in range(H_out):
            for j in range(W_out):
                z[:, :, i, j] = np.tensordot(x[:, :, i:i+self.K, j:j+self.K], self.W, axes=((1,2,3),(1,2,3))) +\
                                self.b.reshape(-1)
        return z

    def backward(self, dLdz):
        N, C_out, H_out, W_out = dLdz.shape
        N, C_in, H_in, W_in = self.x.shape
        assert  C_out == self.C_out and C_in == self.C_in
        # find dLdW
        self.dLdW = np.empty(shape=(C_out, C_in, self.K, self.K), dtype=np.float64)
        for i in range(C_out):
            for j in range(self.K):
                for k in range(self.K):
                    self.dLdW[i, :, j, k] = np.tensordot(self.x[:, :, j:j+H_out, k:k+W_out],
                                                         (dLdz[:, i, :, :])[:, np.newaxis, :, :],
                                                         axes=((0,2,3),(0,2,3))).reshape(-1)
        # find dLdb
        self.dLdb = np.sum(dLdz, axis=(0,2,3))
        # find dLdx
        dLdx = np.zeros(shape=self.x.shape, dtype=np.float64)
        dLdz = np.pad(dLdz, pad_width=((0,0),(0,0),(self.K - 1, self.K - 1), (self.K - 1, self.K - 1)), mode='constant')
        for i in range(C_out):
            for j in range(H_in):
                for k in range(W_in):
                    dLdx[:, :, j, k] += np.tensordot((dLdz[:, i, j:j+self.K, k:k+self.K])[:, np.newaxis, :, :],
                                                     np.flip((self.W[i, :, :, :])[np.newaxis, :, :, :],
                                                     axis=(2,3)), axes=((1,2,3),(0,2,3)))
        return dLdx

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.padding = padding
        self.C_in = in_channels
        self.C_out = out_channels
        self.K = kernel_size
        self.conv2d_stride1 = Conv2d_stride1(self.C_in, self.C_out, self.K, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(downsampling_factor=stride)

    def forward(self, x):
        # pad with zeros
        x = np.pad(x, pad_width=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        # Conv2d forward
        z = self.conv2d_stride1.forward(x)
        # Downsample forward
        z = self.downsample2d.forward(z)
        return z

    def backward(self, dLdz):
        # Downsample backward
        dLdx = self.downsample2d.backward(dLdz)
        # Conv2d backward
        dLdx = self.conv2d_stride1.backward(dLdx)
        # unpad zeros
        if self.padding>0:
            dLdx = dLdx[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dLdx

if __name__ == "__main__":
    # for testing purposes
    N = 10
    C_in = 5
    C_out = 8
    K = 3
    H_in = 5
    W_in = 5

    conv2d_stride1 = Conv2d_stride1(C_in, C_out, K, f1, f2)
    conv2d = Conv2d(C_in, C_out, K, 2, 0, f1, f2)

    # forward
    x = np.random.randint(0,10,(N, C_in, H_in, W_in))
    z = conv2d_stride1.forward(x)
    # backward
    dLdz = np.random.randint(0,10,size=z.shape)
    dLdx = conv2d_stride1.backward(dLdz)

    print("x.shape ", x.shape)
    print("z.shape ", z.shape)

    # print("dLdx ", dLdx)
    print("dLdx.shape ", dLdx.shape)
    print("dLdz.shape ", dLdz.shape)

    # forward
    x = np.random.randint(0,10,(N, C_in, H_in, W_in))
    z1 = conv2d.forward(x)
    # backward
    dLdz1 = np.random.randint(0,10,size=z1.shape)
    dLdx1 = conv2d.backward(dLdz1)

    print("x.shape ", x.shape)
    print("z1.shape ", z1.shape)
    print("dLdx1.shape ", dLdx1.shape)
    print("dLdz1.shape ", dLdz1.shape)
