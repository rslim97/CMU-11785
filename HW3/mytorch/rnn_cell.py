import numpy as np
from .activation import *

class RNNCell:

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        h, d = self.hidden_size, self.input_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        # print("forward x ", x)
        # print("h_prev_t ", h_prev_t)
        h_t =  h_prev_t @ self.W_hh.T + self.b_ih + x @ self.W_ih.T + self.b_hh
        out = self.activation.forward(h_t)
        # print("out ", out)
        return out

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        dz = self.activation.backward(delta, h_t)
        batch_size = delta.shape[0]
        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih += dz.T @ h_prev_l / batch_size
        self.dW_hh += dz.T @ h_prev_t / batch_size
        self.db_ih += np.sum(dz.T, axis=1) / batch_size
        self.db_hh += np.sum(dz.T, axis=1) / batch_size

        # 2) Compute dx, dh_prev_t
        dx = dz @ self.W_ih
        dh_prev_t = dz @ self.W_hh

        return dx, dh_prev_t
