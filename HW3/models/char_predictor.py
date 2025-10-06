import numpy as np
import sys

sys.path.append("mytorch")
from HW3.mytorch.gru_cell import *
from HW3.mytorch.linear import *

"""
To be clear :
1) Big differences between this problem and the RNN Phoneme Classifier are 1) we are only doing inference
(a forward pass) on this network and 2) there is only 1 layer. This means that the forward method in
the CharacterPredictor can be just 2 or 3 lines of code and the inference function can be completed
in less than 10 lines of code.

2) The difference between this (RNN) approach and MLP approach is, in the context of using MLPs to classify  phonemes, a
context window is needed to capture the dependencies between the phoneme that comes before and after the current phoneme.
Meanwhile, when using RNNs the context window is no longer defined explicitly since the RNN models the sequential depen-
dencies among phonemes implicitly (past and future for bidirectional RNNs; only the past, for vanilla RNNs) and it has 
a memory that was built upon previous inputs, so there is a temporal relationship in the way the input is processed.
"""

class CharacterPredictor(object):
    """CharacterPredictor class.

    This is the neural net that will run one timestep of the input
    You only need to implement the forward method of this class.
    This is to test that your GRU Cell implementation is correct when used as a GRU.

    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        """The network consists of a GRU Cell and a linear layer."""
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.rnn = GRUCell(input_dim, hidden_dim)
        self.projection = Linear(hidden_dim, num_classes)
        self.projection.W = np.random.rand(num_classes, hidden_dim)

    def init_rnn_weights(
            self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
    ):
        # DO NOT MODIFY
        self.rnn.init_weights(
            Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh
        )

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """CharacterPredictor forward.

        A pass through one time step of the input

        Input
        -----
        x: (feature_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        logits: (num_classes)
            hidden state at current time-step.

        hnext: (hidden_dim)
            hidden state at current time-step.

        """
        hnext = self.rnn.forward(x, h)
        # self.projection expects input in the form of (batch_size, input_dimension)
        # Therefore, reshape the input of self.projection as (1,-1)
        logits = self.projection.forward(hnext.reshape(1, -1))  # batch_size = 1
        logits = logits.reshape(-1,) # uncomment once code implemented
        return logits, hnext


def inference(net, inputs):
    """CharacterPredictor inference.

    An instance of the class defined above runs through a sequence of inputs to
    generate the logits for all the timesteps.

    Input
    -----
    net:
        An instance of CharacterPredictor.

    inputs: (seq_len, feature_dim)
            a sequence of inputs of dimensions.

    Returns
    -------
    logits: (seq_len, num_classes)
            one per time step of input..

    """

    # This code should not take more than 10 lines.
    seq_len = inputs.shape[0]
    logits = np.empty(shape=(seq_len, net.num_classes), dtype=np.float64)
    h = np.zeros(net.hidden_dim)
    for i in range(seq_len):
        x = inputs[i]
        y, h = net.forward(x, h)
        logits[i] = y
    return logits