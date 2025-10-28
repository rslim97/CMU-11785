from HW1.P1.mytorch.linear import *
from HW1.P1.mytorch.activation import *


class Model:
    def __init__(self):
        self.layers = None

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, dLdx):
        raise NotImplementedError()


class MLP0(Model):

    def __init__(self, debug=False):
        Model.__init__(self)
        self.debug = debug
        self.layers = [Linear(2, 3), ReLU()]

    def forward(self, A0):
        # l = len(self.layers)
        # for i in range(l):
        #     x = self.layers[i].forward(x)
        Z1 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z1)
        if self.debug == True:
            self.Z1 = Z1
            self.A1 = A1
        return A1

    def backward(self, dLdA1):
        # l = len(self.layers)
        # for i in reversed(range(l)):
        #     dLdx = self.layers[i].backward(dLdx)
        dLdZ1 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ1)
        if self.debug == True:
            self.dLdZ1 = dLdZ1
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1(Model):

    def __init__(self, debug=False):
        Model.__init__(self)
        self.debug = debug
        self.layers = [Linear(2, 3), ReLU(), Linear(3, 2), ReLU()]

    def forward(self, x):
        Z1 = self.layers[0].forward(x)
        A1 = self.layers[1].forward(Z1)
        Z2 = self.layers[2].forward(A1)
        A2 = self.layers[3].forward(Z2)
        if self.debug == True:
            self.Z1 = Z1
            self.A1 = A1
            self.Z2 = Z2
            self.A2 = A2
        return A2

    def backward(self, dLdA):
        dLdZ2 = self.layers[3].backward(dLdA)
        dLdA1 = self.layers[2].backward(dLdZ2)
        dLdZ1 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ1)
        if self.debug == True:
            self.dLdZ2 = dLdZ2
            self.dLdA1 = dLdA1
            self.dLdZ1 = dLdZ1
            self.dLdA0 = dLdA0
        return dLdA0
