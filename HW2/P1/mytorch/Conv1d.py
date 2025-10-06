import numpy as np
from .resampling import *

def f1(C_out, C_in, K):
    return np.random.randint(0, 5, size=(C_out, C_in, K))

def f2(C_out):
    return np.random.randint(0, 3, size=(C_out, 1))

class Conv1d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn):
        self.C_in = in_channels
        self.C_out = out_channels
        self.K = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)
        if bias_init_fn is None:
            self.b = np.zeros(shape=(out_channels, 1))
        else:
            self.b = bias_init_fn(out_channels)
        self.dLdW = None
        self.dLdb = None

    def forward(self, x):
        self.x = x  # save input for backprop
        N, C_in, W_in = x.shape
        W_out = (W_in - self.K) + 1
        z = np.empty(shape=(N, self.C_out, W_out), dtype=np.float64)
        for i in range(W_out):
            z[:, :, i] = np.tensordot(x[:, :, i:i + self.K], self.W, axes=((1,2),(1,2))) + self.b.reshape(-1)
        return z

    def backward(self, dLdz):
        N, C_out, W_out = dLdz.shape
        N, C_in, W_in = self.x.shape
        assert C_out == self.C_out and C_in == self.C_in
        # find dLdW
        self.dLdW = np.empty(shape=(C_out, C_in, self.K), dtype=np.float64)
        for i in range(C_out):
            # Convolve input x with dLdz as filter.
            for j in range(self.K):
                # With tensordot, broadcasting is handled implicitly.
                # Size of immediate result after convolution is (C_in, 1) reshape to (C_in,).
                self.dLdW[i, :, j] = np.tensordot(self.x[:, :, j:j+W_out], dLdz[:, i:i+1, :],
                                                  axes=((0,2),(0,2))).reshape(-1)
        # find dLdb
        self.dLdb = np.sum(dLdz, axis=(0, 2))
        # find dLdx
        dLdx = np.zeros(shape=self.x.shape, dtype=np.float64)
        # pad dLdz with K-1 zeros left and right, to gain back K-1 columns loss during forward,
        # i.e. to get back original input size.
        dLdz = np.pad(dLdz, pad_width=((0, 0), (0, 0), (self.K - 1, self.K - 1)), mode='constant')
        for i in range(C_out):
            # Convolve dLdz with 180^o rotated kernel as filter.
            for j in range(W_in):
                # because all outputs are of the same size (N, C_in) we should accumulate the results.
                dLdx[:, :, j] += np.tensordot(dLdz[:, i:i+1, j:j+self.K], np.flip(self.W[i:i+1, :, :],
                                                                                  axis=2), axes=((1,2),(0,2)))
        return dLdx

class Conv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):

        self.stride = stride
        self.padding = padding
        self.C_in = in_channels
        self.C_out = out_channels
        self.K = kernel_size
        self.conv1d_stride1 = Conv1d_stride1(self.C_in, self.C_out, self.K, weight_init_fn, bias_init_fn)
        self.downsampled1d = Downsample1d(downsampling_factor=stride)

    def forward(self, x):
        # pad with zeros
        x = np.pad(x, pad_width=((0, 0),(0, 0),(self.padding, self.padding)), mode='constant')
        # Conv1d forward
        z = self.conv1d_stride1.forward(x)
        # Downsample1d forward
        z = self.downsampled1d.forward(z)
        return z

    def backward(self, dLdz):
        # Downsample1d backward
        dLdx = self.downsampled1d.backward(dLdz)
        # Conv1d backward
        dLdx = self.conv1d_stride1.backward(dLdx)
        # unpad zeros
        if self.padding>0:
            dLdx = dLdx[:, :, self.padding:-self.padding]
        return dLdx



if __name__ == "__main__":
    N = 10
    C_in = 5
    C_out = 8
    K = 3
    W_in = 5

    conv1d_stride1 = Conv1d_stride1(C_in, C_out, K, f1, f2)
    conv1d = Conv1d(C_in, C_out, K, stride=2, padding=0, weight_init_fn=f1, bias_init_fn=f2)

    # forward
    x = np.random.randint(0, 10, size=(N, C_in, W_in))
    z = conv1d.forward(x)

    # backward
    dLdz = np.random.randint(0, 10, size=z.shape)
    dLdx = conv1d.backward(dLdz)

    # print(h)
    print("x.shape", x.shape)
    print("z.shape ",z.shape)
    print("dLdx.shape ", dLdx.shape)
    print("dLdZ.shape", dLdz.shape)
