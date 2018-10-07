# Python implementation of Salakhutdinov and Hinton matlab code in 2006 science paper
# Salakhutdinov and Hinton (2006): https://www.cs.toronto.edu/~hinton/science.pdf
# Supporting Online Material: http://science.sciencemag.org/content/suppl/2006/08/04/313.5786.504.DC1

# goal: learn weights/biases in the stacked rbm, save these, to be used in rbm_matlab_mnist_result.py
import numpy as np
import pandas as pd
import math
import mnist
from rbm_matlab_mnist_special import RBM
import os

# set hyperparameters
maxepoch = 1
numhid = 1000
numdims = 784
numpen = 500
numpen2 = 250
numopen = 30

# read data
x_train, t_train, x_test, t_test = mnist.load()

# scale data
x_train = x_train/255
x_test = x_test/255

# define a function that groups input data into batches
# code borrowed from andrew course 2 week 6
def random_mini_batches(X, mini_batch_size=10, seed=0):
    """
    Creates a list of random minibatches from input X

    Arguments:
    X -- input data, of shape (number of examples, input_size)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous mini_batch
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled = X[permutation, :]

    # Step 2: Partition shuffled. Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch = shuffled[k * mini_batch_size: (k + 1) * mini_batch_size, :]
        ### END CODE HERE ###
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch = shuffled[math.floor(m / mini_batch_size) * mini_batch_size: m, :]

        ### END CODE HERE ###
        mini_batches.append(mini_batch)

    return mini_batches

def _logistic(x):
    return 1.0 / (1 + np.exp(-x))

# batch input data
batchdata = random_mini_batches(x_train, mini_batch_size=100)  # list of batches of input data

if __name__ == "__main__":
    jun = RBM(numhid, batchdata, maxepoch, numdims)
    batchposhidprobs, vishid, visbiases, hidbiases, error = jun.rbm()
    hidrecbiases = hidbiases
    home = os.getenv('HOME')
    np.save(home + '/Deep_Learning_Examples/RBM/vishid.npy', vishid)
    np.save(home + '/Deep_Learning_Examples/RBM/hidrecbiases.npy', hidrecbiases)
    np.save(home + '/Deep_Learning_Examples/RBM/visbiases.npy', visbiases)

    batchdata = batchposhidprobs
    numhid = numpen
    numdims = batchdata[0].shape[1]
    jun = RBM(numhid, batchdata, maxepoch, numdims)
    batchposhidprobs, vishid, visbiases, hidbiases, error = jun.rbm()
    hidpen = vishid
    penrecbiases = hidbiases
    hidgenbiases = visbiases
    np.save(home + '/Deep_Learning_Examples/RBM/hidpen.npy', hidpen)
    np.save(home + '/Deep_Learning_Examples/RBM/penrecbiases.npy', penrecbiases)
    np.save(home + '/Deep_Learning_Examples/RBM/hidgenbiases.npy', hidgenbiases)

    batchdata = batchposhidprobs
    numhid = numpen2
    numdims = batchdata[0].shape[1]
    jun = RBM(numhid, batchdata, maxepoch, numdims)
    batchposhidprobs, vishid, visbiases, hidbiases, error = jun.rbm()
    hidpen2 = vishid
    penrecbiases2 = hidbiases
    hidgenbiases2 = visbiases
    np.save(home + '/Deep_Learning_Examples/RBM/hidpen2.npy', hidpen2)
    np.save(home + '/Deep_Learning_Examples/RBM/penrecbiases2.npy', penrecbiases2)
    np.save(home + '/Deep_Learning_Examples/RBM/hidgenbiases2.npy', hidgenbiases2)

    batchdata = batchposhidprobs
    numhid = numopen
    numdims = batchdata[0].shape[1]
    jun = RBM(numhid, batchdata, maxepoch, numdims)
    batchposhidprobs, vishid, visbiases, hidbiases, error = jun.rbmhidlinear()
    hidtop = vishid
    toprecbiases = hidbiases
    topgenbiases = visbiases
    np.save(home + '/Deep_Learning_Examples/RBM/hidtop.npy', hidtop)
    np.save(home + '/Deep_Learning_Examples/RBM/toprecbiases.npy', toprecbiases)
    np.save(home + '/Deep_Learning_Examples/RBM/topgenbiases.npy', topgenbiases)