import sys
sys.path.append('P1/models')
sys.path.append('P1/mytorch')
from model import Model
from linear import *
from activation import *
from batchnorm import BatchNorm1d
from loss import *
from sgd import SGD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class myFirstMLP(Model):
    def __init__(self):
        super().__init__()
        self.layers = [Linear(1, 32), ReLU(), Linear(32, 1), Identity()]

    def forward(self, x):
        l = len(self.layers)
        for i in range(l):
            x = self.layers[i].forward(x)
        return x

    def backward(self, dLdx):
        l = len(self.layers)
        for i in reversed(range(l)):
            dLdx = self.layers[i].backward(dLdx)
        return dLdx

class mySecondMLP(Model):
    def __init__(self):
        super().__init__()
        self.layers = [Linear(1, 10),
                       ReLU(),
                       Linear(10, 24),
                       ReLU(),
                       Linear(24, 1),
                       Identity()]

    def forward(self, x):
        l = len(self.layers)
        for i in range(l):
            x = self.layers[i].forward(x)
        return x

    def backward(self, dLdx):
        l = len(self.layers)
        for i in reversed(range(l)):
            dLdx = self.layers[i].backward(dLdx)
        return dLdx

class myThirdMLP(Model):
    def __init__(self):
        super().__init__()
        self.layers = [Linear(1, 10),
                       ReLU(),
                       Linear(10, 24),
                       BatchNorm1d(24),
                       ReLU(),
                       Linear(24, 1),
                       Identity()]

    def forward(self, x):
        l = len(self.layers)
        for i in range(l):
            x = self.layers[i].forward(x)
        return x

    def backward(self, dLdx):
        l = len(self.layers)
        for i in reversed(range(l)):
            dLdx = self.layers[i].backward(dLdx)
        return dLdx

def get_batches(dataset, batch_size):
    # x = dataset[0, :].reshape(-1, 1)
    # y = dataset[1, :].reshape(-1, 1)
    x, y = dataset
    # print("x ",x)
    N = x.shape[0]
    # print(N)
    # shuffle at the start of epoch
    indices = np.arange(N)
    np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        batch_index = indices[start:end]
        # print(batch_index)
        # print("x[batch_index] :", x[batch_index])
        # print("y[batch_index] :", y[batch_index])
        yield x[batch_index], y[batch_index]

def normalize(x):
    N = x.shape[0]
    mu = np.sum(x, axis=0, keepdims=True) / N
    var = np.sum((x - mu) * (x - mu), axis=0, keepdims=True) / N
    x[:] = (x - mu) / np.sqrt(var + 1e-8)  # modify in-place

# def to_column(x):
#     return x.reshape(-1, 1)

if __name__ == "__main__":
    x = np.linspace(0, 20, 100)
    y = np.exp(-1) * np.sin(x) + np.random.randn(len(x)) / 10
    # add C=1 dimension to x and y of shape (N,)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # normalize inputs and outputs. Without normalizing the network, it would be harder to fit later parts of sinusoid.
    normalize(x)
    normalize(y)
    # print("x ", x)
    # print("y ", y)
    print("x.shape ", x.shape)
    print("y.shape ", y.shape)
    dataset = (x, y)

    mlp = myThirdMLP()
    mseloss = MSELoss()

    # # initialize weights for myFirstMLP
    # W1 = np.random.randn(32, 1).astype("f")
    # b1 = np.random.randn(32).astype("f")
    # W2 = np.random.randn(1, 32).astype("f")
    # b2 = np.random.randn(1).astype("f")
    # mlp.layers[0].W = W1
    # mlp.layers[0].b = b1.reshape(-1, 1)
    # mlp.layers[2].W = W2
    # mlp.layers[2].b = b2.reshape(-1, 1)

    # # initialize weights for mySecondMLP
    # W1 = np.random.randn(10, 1).astype("f")
    # b1 = np.random.randn(10).astype("f")
    # W2 = np.random.randn(24, 10).astype("f")
    # b2 = np.random.randn(24).astype("f")
    # W3 = np.random.randn(1, 24).astype("f")
    # b3 = np.random.randn(1).astype("f")
    # mlp.layers[0].W = W1
    # mlp.layers[0].b = b1.reshape(-1, 1)
    # mlp.layers[2].W = W2
    # mlp.layers[2].b = b2.reshape(-1, 1)
    # mlp.layers[4].W = W3
    # mlp.layers[4].b = b3.reshape(-1, 1)

    # initialize weights for mySecondMLP
    W1 = np.random.randn(10, 1).astype("f")
    b1 = np.random.randn(10).astype("f")
    W2 = np.random.randn(24, 10).astype("f")
    b2 = np.random.randn(24).astype("f")
    W3 = np.random.randn(1, 24).astype("f")
    b3 = np.random.randn(1).astype("f")
    mlp.layers[0].W = W1
    mlp.layers[0].b = b1.reshape(-1, 1)
    mlp.layers[2].W = W2
    mlp.layers[2].b = b2.reshape(-1, 1)
    mlp.layers[5].W = W3
    mlp.layers[5].b = b3.reshape(-1, 1)

    optimizer = SGD(mlp, lr=0.01)
    losses = []
    n_epochs = 2000
    batch_size = 10
    # save outputs for animation
    outputs_per_epoch = []
    for _ in range(n_epochs):
        # mean_loss = 0
        train_batch_loss = []
        for x_batch, y_batch in get_batches(dataset, batch_size):
            # print("x_batch ", x_batch.shape)
            # print("y_batch ", y_batch.shape)
            # get nn output
            # print("Before update:", mlp.layers[0].W)
            out = mlp.forward(x_batch)

            loss = mseloss.forward(out, y_batch)
            # print("loss :", loss)
            dLdx = mseloss.backward()
            # print("dLdx main:", dLdx)
            mlp.backward(dLdx)
            optimizer.step()
            # print("After update:", mlp.layers[0].W)

            # mean_loss += loss
            train_batch_loss.append(loss)
        # print("x.shape ", x.shape)
        outputs = mlp.forward(x)

        outputs_per_epoch.append(outputs)
        losses.append(np.mean(train_batch_loss))
        # losses.append(mean_loss/ (x.shape[0] / batch_size))


    # """ Plot results """
    # # test case
    # y_test = y.flatten() + np.random.randn(len(x)) / 5
    # output_test = mlp.forward(x)
    #
    # fig, (ax1) = plt.subplots(1)
    # fig.suptitle('Axes values are scaled individually by default')
    # ax1.plot(x, y_test)
    # ax1.plot(x, output_test, 'r')
    # # ax2.plot(np.arange(n_epochs), losses)
    # plt.show()

    """ Animate """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_aspect('equal')
    # fig.suptitle('one-hidden-layer-MLP : [1, 32, 1]')
    # fig.suptitle('two-hidden-layer-MLP : [1, 10, 24, 1]')
    # fig.suptitle('two-hidden-layer-MLP with BatchNorm : [1, 10, 24, BN, 1]')
    step = 10
    frames = list(range(0, n_epochs, step))

    def animate(i):
        ax1.clear()
        ax2.clear()
        ax1.plot(x, y, "*")  # gt
        ax1.plot(x, outputs_per_epoch[i])
        # ax1.set_title('one-hidden-layer-MLP : [1, 32, 1]')
        # ax1.set_title('two-hidden-layer-MLP : [1, 10, 24, 1]')
        ax1.set_title('two-hidden-layer-MLP with BatchNorm\n [1, 10, 24, BN, 1]')
        ax2.text(0.5, 0.5, f"epoch: {i+step}", verticalalignment='center', horizontalalignment='center',
                 transform=ax2.transAxes)
        ax2.plot(np.arange(i+step), losses[0:i+step], 'tab:blue')
        ax2.set_title('Training loss')
        # line3, = ax2.plot(x, ou)
        # return line, line2, text
        return []

    ani = FuncAnimation(fig, animate, frames=frames, interval=50, blit=False, repeat=False)
    ani.save("P1/gif/animation1.gif", dpi=300, writer=PillowWriter(fps=15))
    plt.show()

"""
1. Problem : MLP cannot fit later parts of the since function i.e. only fits the first half.
   Solution : For the NN learn the sine function well, scale the input values in the [-1, +1] range 
   (neural networks don't like big values).
2. Problem : MLP cannot fit function well.
   Solution : Use more neurons (wider) and make deeper (for more expressiveness). Use Relu instead of tanh to solve for
   vanishing gradients during backprop (better learning).
3. Problem : Using batchnorm seems like no improvement in learning.
   Solution : Use higher learning rates, batchnorm allows higher learning rate.
"""