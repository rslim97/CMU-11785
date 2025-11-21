import numpy as np
from .activation import *


class GRUCell:

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.n_act = Tanh()

        self.r = None
        self.z = None
        self.n = None
        self.n4 = None

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
        """
        self.x = x
        self.hidden = h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        r1 = self.Wrx @ x
        r2 = r1 + self.brx
        r3 = self.Wrh @ h_prev_t
        r4 = r3 + self.brh
        r5 = r2 + r4
        self.r = self.r_act.forward(r5)

        assert self.r.shape == (self.h,)

        z1 = self.Wzx @ x
        z2 = z1 + self.bzx
        z3 = self.Wzh @ h_prev_t
        z4 = z3 + self.bzh
        z5 = z2 + z4
        self.z = self.z_act.forward(z5)

        assert self.z.shape == (self.h,)

        n1 = self.Wnx @ x
        n2 = n1 + self.bnx
        n3 = self.Wnh @ h_prev_t
        n4 = n3 + self.bnh
        self.n4 = n4
        n5 = self.r * n4
        n6 = n2 + n5
        self.n = self.n_act.forward(n6)

        assert self.n.shape == (self.h,)

        h1 = 1 - self.z
        h2 = self.n * h1
        h3 = self.z * h_prev_t
        h_t = h2 + h3

        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        delta = delta.squeeze()
        dLdn_t = delta * (1 - self.z)
        dLdn_6 = self.n_act.backward(dLdn_t)
        dLdn_4 = dLdn_6 * self.r
        self.dbnh = dLdn_4
        self.dWnh = dLdn_4[:, np.newaxis] @ self.hidden[np.newaxis, :]
        self.dbnx = dLdn_6
        self.dWnx = dLdn_6[:, np.newaxis] @ self.x[np.newaxis, :]

        dLdh1 = delta * self.n
        dLdz_t = -dLdh1 + delta * self.hidden
        dLdz_5 = self.z_act.backward(dLdz_t)
        self.dbzh = dLdz_5
        self.dWzh = dLdz_5[:, np.newaxis] @ self.hidden[np.newaxis, :]
        self.dbzx = dLdz_5
        self.dWzx = dLdz_5[:, np.newaxis] @ self.x[np.newaxis, :]

        if self.n4 is not None:
            dLdr_t = dLdn_6 * self.n4
        else:
            dLdr_t = 0
        dLdr_5 = self.r_act.backward(dLdr_t)
        self.dbrh = dLdr_5
        self.dWrh = dLdr_5[:, np.newaxis] @ self.hidden[np.newaxis, :]
        self.dbrx = dLdr_5
        self.dWrx = dLdr_5[:, np.newaxis] @ self.x[np.newaxis, :]

        dx = dLdr_5[np.newaxis, :] @ self.Wrx + \
            dLdz_5[np.newaxis, :] @ self.Wzx + \
            dLdn_6[np.newaxis, :] @ self.Wnx

        dh_prev_t = dLdr_5[np.newaxis, :] @ self.Wrh + \
            dLdz_5[np.newaxis, :] @ self.Wzh + \
            dLdn_4[np.newaxis, :] @ self.Wnh + \
            delta[np.newaxis, :] * self.z[np.newaxis, :]

        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
