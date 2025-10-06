import matplotlib.colors
import sys
sys.path.append('P1/models')
sys.path.append('P1/mytorch')
from model import Model
from linear import *
from activation import *
from loss import *
from sgd import SGD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class multiClassModel1(Model):
    def __init__(self):
        super().__init__()
        # init layers, assume a task of classifying 3 classes
        self.layers = [Linear(2, 10), ReLU(), Linear(10, 3)]

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

class multiClassModel2(Model):
    def __init__(self):
        super().__init__()
        # init layers, assume a task of classifying 3 classes
        self.layers = [Linear(2, 32),
                       ReLU(),
                       Linear(32, 24),
                       ReLU(),
                       Linear(24, 3)]

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


# note: no Softmax layer are used in the models, since Softmax is already included in crossEntropyLoss.
# The model outputs raw logits, which need to be passed through a Softmax during evaluation
# to get probabilities.

def get_batches(dataset, batch_size):
    x, y = dataset
    N = x.shape[0]
    # shuffle at the start of epoch
    indices = np.arange(N)
    np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        batch_index = indices[start:end]
        yield x[batch_index], y[batch_index]

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x


if __name__ == "__main__":
    np.random.seed(42)
    n_classes = 3
    n_samples = 100
    x_list = []
    y_list = []

    for i in range(n_classes):
        """ using 2D data samples for ease of visualization """
        # use box-mueller transform to generate normal gaussian distribution
        u1 = np.random.random(size=(n_samples))  # generate random uniform between 0 and 1
        u2 = np.random.random(size=(n_samples))
        R = np.sqrt(-2 * np.log(u1))
        theta = 2 * np.pi * u2
        z0 = R * np.cos(theta)
        z1 = R * np.sin(theta)
        z = np.hstack((z0[..., np.newaxis], z1[...,np.newaxis]))
        mu = np.random.normal(3 * i * np.random.randn(),5,size=(1, 2))
        sigma = 2.5
        x_ = sigma * z + mu
        y_ = np.zeros(shape=(n_samples, n_classes))
        y_[:, i] = 1
        x_list.append(x_)
        y_list.append(y_)

        # ax1.plot(x[i][:, 0], x[i][:, 1], "o", color=y[i][0, :])

    x = np.vstack(x_list)  # convert list to array
    y = np.vstack(y_list)
    dataset = (x, y)
    # initialize model and loss function
    mlp = multiClassModel2()
    cross_entropy_loss = CrossEntropyLoss()

    # # initialize weights for model 1
    # W1 = np.random.randn(10, 2).astype("f")
    # b1 = np.random.randn(10).astype("f")
    # W2 = np.random.randn(3, 10).astype("f")
    # b2 = np.random.randn(3).astype("f")
    # mlp.layers[0].W = W1
    # mlp.layers[0].b = b1.reshape(-1, 1)
    # mlp.layers[2].W = W2
    # mlp.layers[2].b = b2.reshape(-1, 1)

    # initialize weights for model 2
    W1 = np.random.randn(32, 2).astype("f")
    b1 = np.random.randn(32).astype("f")
    W2 = np.random.randn(24, 32).astype("f")
    b2 = np.random.randn(24).astype("f")
    W3 = np.random.randn(3, 24).astype("f")
    b3 = np.random.randn(3).astype("f")
    mlp.layers[0].W = W1
    mlp.layers[0].b = b1.reshape(-1, 1)
    mlp.layers[2].W = W2
    mlp.layers[2].b = b2.reshape(-1, 1)
    mlp.layers[4].W = W3
    mlp.layers[4].b = b3.reshape(-1, 1)

    # initialize optimizer
    optimizer = SGD(mlp)
    losses = []
    n_epochs = 1000
    batch_size = 50
    # save outputs for animation
    outputs_per_epoch = []
    for _ in range(n_epochs):
        train_batch_loss = []
        for x_batches, y_batches in get_batches(dataset, batch_size):
            # reset gradients

            # forward
            # print("x_batches.shape ",x_batches.shape)
            out = mlp.forward(x_batches)
            loss = cross_entropy_loss.forward(out, y_batches)
            # backward
            dLdx = cross_entropy_loss.backward()
            mlp.backward(dLdx)
            # update weights
            optimizer.step()
            train_batch_loss.append(loss)
        outputs = mlp.forward(x)
        probs = softmax(outputs)
        outputs_per_epoch.append(probs)
        losses.append(np.mean(train_batch_loss))

    print("x.shape ",x.shape)

    # # test on a single training data
    # y_test = y[[98]]
    # x_test = x[[98]]
    # out_test = mlp.forward(x_test)
    # probs = softmax(out_test)
    # print("out_test ", out_test)
    # print("y_test ", y_test)
    # print("probs ", probs)
    # fig, (ax1) = plt.subplots(1)
    # ax1.plot(x_test[0, 0], x_test[0, 1], "o", color=y_test)

    # test on all training data
    y_test = y  # gt
    x_test = x

    # out_test = mlp.forward(x_test)
    # probs = softmax(out_test)
    # # print("out_test ", out_test)
    # # print("y_test ", y_test)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # ax1.set_aspect('equal')
    # ax2.set_aspect('equal')
    # for i in range(len(x_test)):
    #     # plot gt
    #     ax1.set_title('Ground truth data')
    #     ax1.plot(x_test[i, 0], x_test[i, 1], "o", color=y_test[i])
    #     # plot prediction
    #     # ax2.set_title('one-hidden-layer-MLP [2, 10, 3] prediction')
    #     ax2.set_title('two-hidden-layer-MLP [2, 24, 10, 3] prediction')
    #     ax2.plot(x_test[i, 0], x_test[i, 1], "o", color=probs[i])
    #
    # plt.show()

    """ Animate """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    step = 10
    frames = list(range(0, n_epochs, step))
    # print(out)
    def animate(i):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.set_title('Ground truth data')
        ax2.set_title('two-hidden-layer-MLP prediction\n [2, 32, 24, 3] ')
        ax3.set_title('Training loss')
        ax3.text(0.5, 0.5, f"epoch: {i+step}", verticalalignment='center', horizontalalignment='center',
                 transform=ax3.transAxes)
        cmap = matplotlib.colormaps['tab10']
        for j in range(len(x_test)):
            # print("y_test[j] ", y_test[j])
            ax1.plot(x_test[j, 0], x_test[j, 1], "o", color=cmap(np.argmax(y_test[j])))
            ax2.plot(x_test[j, 0], x_test[j, 1], "o", color=cmap(np.argmax(outputs_per_epoch[i][j])))
            ax3.plot(np.arange(i+step), losses[0:i+step], 'tab:blue')
        return []

    ani = FuncAnimation(fig, animate, frames=frames, interval=50, blit=False, repeat=False)
    ani.save("P1/gif/animation2.gif", dpi=300, writer=PillowWriter(fps=15))
    plt.show()
