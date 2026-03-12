import numpy as np

from HW3.P1.mytorch.rnn_cell import *
from HW3.P1.mytorch.linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO: Understand then uncomment this code :)
        self.rnn = [
            (
                RNNCell(input_size, hidden_size)
                if i == 0
                else RNNCell(hidden_size, hidden_size)
            )
            for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps."""
        N, T, D = x.shape
        L = self.num_layers
        H = self.hidden_size

        if h_0 is None:
            hiddens = np.zeros((L, N, H), dtype=np.float64)
        else:
            L, _, H = h_0.shape
            assert L == self.num_layers, H == self.hidden_size
            hiddens = h_0
        self.hiddens.append(hiddens.copy())
        self.x = x
        for t in range(T):
            for l in range(L):
                # Compute h_prev_l
                if l == 0:
                    h_prev_l = x[:, t, :]
                else:
                    h_prev_l = hiddens[l - 1, :, :]
                # Compute h_prev_t
                h_prev_t = hiddens[l]
                # Perform forward on RNN Cell
                h_next = self.rnn[l].forward(h_prev_l, h_prev_t)
                # Update hiddens for next time step
                hiddens[l] = h_next

            self.hiddens.append(hiddens.copy())

        logits = self.output_layer.forward(h_next)
        return logits

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).
        N: number of samples or batch size
        L: number of layers
        H: hidden dimension
        D: input dimension
        H_out: output dimension
        delta: Upstream gradient, dL/dy from linear layer before loss function
        i.e. in last RNN Cell, h_next -> y -> L: (N, H_out)
        self.hiddens: Contains h_next computed in each RNN Cell: (T+1, L, N, H)
        self.x: Input sequence: (N, T, D)
        dh_next: Contains upstream gradients: (L, N, H)
        for backprop, cache = (h_next, h_prev_l, h_prev_t, dh_next)
        """
        T = self.x.shape[1]
        L = self.num_layers
        H = self.hidden_size
        N, _ = delta.shape
        dh_next = np.zeros((L, N, H), dtype=np.float64)
        # Compute upstream gradient, dL/dh_next, for last RNN Cell
        dh_next[-1] = self.output_layer.backward(delta)
        # self.output_layer.backward(delta)
        for t in reversed(range(T)):
            for l in reversed(range(L)):
                # Compute variables in cache need for backprop
                h_next = self.hiddens[t + 1][l, :, :]
                h_prev_t = self.hiddens[t][l, :, :]
                # h_prev_l can come from either input sequence or h_next from previous cell, l-1
                if l == 0:
                    h_prev_l = self.x[:, t, :]
                else:
                    h_prev_l = self.hiddens[t + 1][l - 1, :, :]
                # Done computing variables in cache needed for backprop
                # Perform backprop on RNN Cell
                h_prev_l, h_prev_t = self.rnn[l].backward(
                    dh_next[l], h_next, h_prev_l, h_prev_t
                )
                # Update dh_next for the next previous time step
                dh_next[l] = h_prev_t
                # Add downstream gradient to the upstream gradient of the next previous cell step
                if l != 0:
                    dh_next[l - 1] += h_prev_l
        return dh_next / N


if __name__ == "__main__":
    input_size = 3
    hidden_size = 4
    output_size = 5
    rnn = RNNPhonemeClassifier(input_size, hidden_size, output_size)

    N = 10
    seq_len = 3
    x = np.random.random((N, seq_len, input_size))
    logits = rnn.forward(x)
    # print(logits.shape)
    # print(logits)
    delta = np.random.random((N, output_size))
    dh = rnn.backward(delta)
    print(dh)
    print(dh.shape)
