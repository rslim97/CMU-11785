import numpy as np
from .resampling import *
from .Conv1d import *
from .Conv2d import *

"""
Tranpose convolution is also called fractionally strided convolution.
"""

class ConvTranspose1d:

    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn, bias_init_fn):

        self.K = kernel_size
        self.S = upsampling_factor  # factor S
        self.C_in = in_channels
        self.C_out = out_channels

        self.upsampling1d = Upsample1d(upsampling_factor=upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)

    def forward(self, x):
        # Upsampling1d forward
        z = self.upsampling1d.forward(x)
        # Conv forward
        z = self.conv1d_stride1.forward(z)
        return z

    def backward(self, dLdz):
        # Conv backward
        dLdx = self.conv1d_stride1.backward(dLdz)
        # Upsampling1d backward
        dLdx = self.upsampling1d.backward(dLdx)
        return dLdx

class ConvTranspose2d:
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn, bias_init_fn):

        self.K = kernel_size
        self.S = upsampling_factor
        self.C_in = in_channels
        self.C_out = out_channels

        self.upsampling2d = Upsample2d(upsampling_factor=upsampling_factor)
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)

    def forward(self, x):
        # Upsampling2d forward
        z = self.upsampling2d.forward(x)
        # Conv forward
        z = self.conv2d_stride1.forward(z)
        return z

    def backward(self, dLdz):
        # Conv backward
        dLdx = self.conv2d_stride1.backward(dLdz)
        # Upsampling2d backward
        dLdx = self.upsampling2d.backward(dLdx)
        return dLdx