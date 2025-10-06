import numpy as np

class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.dLdgamma = np.zeros((1, num_features))
        self.dLdbeta = np.zeros((1, num_features))

        self.running_mu = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The Eval_parameter is to indicate whether we are in the training phase
        of the problem or the inference phase. So see what values you need
        to recompute when eval is False.
        ZN : Normalized input data, Z
        ZB : Scaled and transformed Z
        """
        self.Z = Z
        self.N = Z.shape[0]
        ones = np.ones((self.N, 1))
        self.mu = ones.T @ (Z / self.N)
        self.var = ones.T @ (((Z - ones @ self.mu) ** 2) / self.N)

        if eval == False:
            # training mode
            self.ZN = (Z - ones @ self.mu) / (ones @ np.sqrt(self.var + self.eps))
            self.ZB = self.ZN * (ones @ self.gamma) + ones @ self.beta

            self.running_mu = self.alpha * self.running_mu + (1 - self.alpha) * self.mu
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        else:
            # inference mode
            self.ZN = (Z - ones @ self.running_mu) / (ones @ np.sqrt(self.running_var + self.eps))
            self.ZB = self.ZN * (ones @ self.gamma) + ones @ self.beta

        return self.ZB

    def backward(self, dLdZB):
        self.N = dLdZB.shape[0]
        ones = np.ones(shape=(self.N, 1))

        self.dLdgamma = ones.T @ (dLdZB * self.ZN)
        self.dLdbeta = ones.T @ dLdZB
        dLdZN = dLdZB * (ones @ self.gamma)

        dZNdmu = -1 / np.sqrt(ones @ self.var + self.eps)
        dLdmu = ones.T @ (dLdZN * dZNdmu)

        dZNdvar = -0.5 * (self.Z - ones @ self.mu) * np.power(ones @ self.var + self.eps, -1.5)
        dLdvar = ones.T @ (dLdZN * dZNdvar)

        dZNdZ = ones @ (1 / np.sqrt(self.var + self.eps))
        dmudZ = 1 / self.N
        dvardZ = 2 * (self.Z - ones @ self.mu) / self.N
        term1 = dLdZN * dZNdZ
        term2 = (ones @ dLdmu) * dmudZ
        term3 = (ones @ dLdvar) * dvardZ

        dLdZ = term1 + term2 +term3
        return dLdZ
