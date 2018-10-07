# goal: use weights/biases learned in rbm_matlab_minst_general.py to get two things:
# 1. the low dimensional representation of each example
# 2. the reconstruction of each example
# this is basically going to be a translation of the non-backprop part in backprop.m

import numpy as np
import mnist
from rbm_matlab_mnist_general import random_mini_batches
from pylab import imshow, cm, show
import os

## load mnist data, make batches
# read data
x_train, t_train, x_test, t_test = mnist.load()

# scale data
x_train = x_train/255
x_test = x_test/255

# batch input data
batchdata = random_mini_batches(x_train, mini_batch_size=100) # list of batches of input data
numbatches = len(batchdata)

## load weights
home = os.getenv('HOME')
vishid = np.load(home + '/Deep_Learning_Examples/RBM/vishid.npy')
hidrecbiases = np.load(home + '/Deep_Learning_Examples/RBM/hidrecbiases.npy')
visbiases = np.load(home + '/Deep_Learning_Examples/RBM/visbiases.npy')
hidpen = np.load(home + '/Deep_Learning_Examples/RBM/hidpen.npy')
penrecbiases = np.load(home + '/Deep_Learning_Examples/RBM/penrecbiases.npy')
hidgenbiases = np.load(home + '/Deep_Learning_Examples/RBM/hidgenbiases.npy')
hidpen2 = np.load(home + '/Deep_Learning_Examples/RBM/hidpen2.npy')
penrecbiases2 = np.load(home + '/Deep_Learning_Examples/RBM/penrecbiases2.npy')
hidgenbiases2 = np.load(home + '/Deep_Learning_Examples/RBM/hidgenbiases2.npy')
hidtop = np.load(home + '/Deep_Learning_Examples/RBM/hidtop.npy')
toprecbiases = np.load(home + '/Deep_Learning_Examples/RBM/toprecbiases.npy')
topgenbiases = np.load(home + '/Deep_Learning_Examples/RBM/topgenbiases.npy')

## do array concatenation, define w1 through w8
w1 = np.concatenate((vishid, hidrecbiases), axis=0)
w2 = np.concatenate((hidpen, penrecbiases), axis=0)
w3 = np.concatenate((hidpen2, penrecbiases2), axis=0)
w4 = np.concatenate((hidtop, toprecbiases), axis=0)
w5 = np.concatenate((np.transpose(hidtop), topgenbiases), axis=0)
w6 = np.concatenate((np.transpose(hidpen2), hidgenbiases2), axis=0)
w7 = np.concatenate((np.transpose(hidpen), hidgenbiases), axis=0)
w8 = np.concatenate((np.transpose(vishid), visbiases), axis=0)

## define logistic function
def _logistic(x):
    return 1.0 / (1 + np.exp(-x))

## calculate the low dimensional vector for each example
w4_list = [] #use this list to collect w4probs for all batches, each element in this list is a 100 x 30 array
dataout_list = [] #use this list to collect dataout for all batches, each element in this list is a 100 x 784 array
err = 0 #use this number to record the cumulative sum of mean squared errors between data and reconstructed data, where the cum sum is over all batches
for batch in range(numbatches):
    # extract one batch of data
    data = batchdata[batch]

    # number of examples in this batch
    numcases = data.shape[0]

    # concatenate data with string of ones
    data = np.concatenate((data, np.ones((numcases, 1))), axis=1)

    # calculate hidden units, one layer at a time
    w1probs = _logistic((np.dot(data, w1))) #100 x 1000
    w1probs = np.concatenate((w1probs, np.ones((numcases, 1))), axis=1)
    w2probs = _logistic((np.dot(w1probs, w2))) #100 x 500
    w2probs = np.concatenate((w2probs, np.ones((numcases, 1))), axis=1)
    w3probs = _logistic((np.dot(w2probs, w3))) #100 x 250
    w3probs = np.concatenate((w3probs, np.ones((numcases, 1))), axis=1)
    w4probs = np.dot(w3probs, w4) #100 x 30
    w4_list.append(w4probs)
    w4probs = np.concatenate((w4probs, np.ones((numcases, 1))), axis=1)

    # keep going
    w5probs = _logistic((np.dot(w4probs, w5))) #100 x 250
    w5probs = np.concatenate((w5probs, np.ones((numcases, 1))), axis=1)
    w6probs = _logistic((np.dot(w5probs, w6))) #100 x 500
    w6probs = np.concatenate((w6probs, np.ones((numcases, 1))), axis=1)
    w7probs = _logistic((np.dot(w6probs, w7))) #100 x 1000
    w7probs = np.concatenate((w7probs, np.ones((numcases, 1))), axis=1)
    dataout = _logistic((np.dot(w7probs, w8))) #100 x 784
    dataout_list.append(dataout)
    err = err + (1/numcases)*(((data[:,:-1]-dataout)**2).sum()) #err is cumulative sum of mean squared error, where the mean is over all cases in a batch, and cum sum is over all batches

# use my hopfield net code to display images
def display(pattern):
    '''
    visualize a pattern as an image
    '''
    imshow(pattern.reshape((28,28)),cmap=cm.binary, interpolation='nearest')
    show()

display(data[:,:-1][0])
display(dataout[0])
