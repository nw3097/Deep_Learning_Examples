# define RBM class here, define rbm method and rbmhidlinear method, class and methods to be called by rbm_matlab_mnist_general.py
import numpy as np
import pandas as pd
import math

def _logistic(x):
    return 1.0 / (1 + np.exp(-x))

class RBM:

    def __init__(self, numhid, batchdata, maxepoch, numdims):
        '''
        numhid: number of hidden units
        batchdata: batchdata as a list, each element is a batch
        maxepoch: number of epochs
        numdims: number of visible units
        '''
        self.numhid = numhid
        self.batchdata = batchdata
        self.maxepoch = maxepoch
        self.numdims = numdims

    def rbm(self, epsilonw = 0.1, epsilonvb = 0.1, epsilonhb = 0.1, weightcost = 0.0002, initialmomentum = 0.5, finalmomentum = 0.9):

        # initialize weights and biases
        vishid = 0.1*np.random.normal(size = (self.numdims, self.numhid)) # v to h weights, 784 x 1000
        hidbiases  = np.zeros((1,self.numhid)) # h to bias weights, 1 x 1000
        visbiases  = np.zeros((1,self.numdims)) # v to bias weights, 1 x 784

        # initialize weights and biases increments
        vishidinc = np.zeros((self.numdims, self.numhid))
        hidbiasinc = np.zeros((1, self.numhid))
        visbiasinc = np.zeros((1, self.numdims))

        # initialize batchposhidprobs
        batchposhidprobs = []

        # initialize error
        error = []

        # iterate over epochs and batches
        for epoch in range(self.maxepoch):
            errsum=0

            for batch in range(len(self.batchdata)):
                # extract one batch of data
                data = self.batchdata[batch]

                # number of examples in this batch
                numcases = data.shape[0]

                # start positive phase
                poshidprobs = _logistic((np.dot(data, vishid) + hidbiases)) # numcases x numhid, 100 x 1000; broadcast hidbiases
                batchposhidprobs.append(poshidprobs)  # put a particular batch's hidden unit probabilites in the right spot
                posprods = np.dot(data.T, poshidprobs)  # numdims x numhid, 784 x 1000
                poshidact = np.sum(poshidprobs, axis = 0)  # (numhid,), (1000,)
                posvisact = np.sum(data, axis = 0) # (numdims,), (784,)

                # end of positive phase
                poshidstates = poshidprobs > np.random.rand(numcases, self.numhid)

                # start negative phase
                negdata = _logistic((np.dot(poshidstates, vishid.T) + visbiases))  # numcases x numdims, 100 x 784
                neghidprobs = _logistic(np.dot(negdata, vishid) + hidbiases)  # numcases x numhid, 100 x 1000
                negprods = np.dot(negdata.T, neghidprobs)  # numdims x numhid, 784 x 1000
                neghidact = np.sum(neghidprobs, axis = 0)  # (numhid,), (1000,)
                negvisact = np.sum(negdata, axis = 0)  # (numdims,), (784,)

                # end of negative phase
                err = np.sum((data - negdata) ** 2)
                errsum = err + errsum
                error.append(errsum)

                if epoch > 5:
                    momentum = finalmomentum
                else:
                    momentum = initialmomentum

                # update weights and biases
                vishidinc = momentum*vishidinc + epsilonw*( (posprods-negprods)/numcases - weightcost*vishid)
                visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact)
                hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact)
                vishid = vishid + vishidinc
                visbiases = visbiases + visbiasinc
                hidbiases = hidbiases + hidbiasinc

        return (batchposhidprobs, vishid, visbiases, hidbiases, error)

    def rbmhidlinear(self, epsilonw = 0.001, epsilonvb = 0.001, epsilonhb = 0.001, weightcost = 0.0002, initialmomentum = 0.5, finalmomentum = 0.9):

        # initialize weights and biases
        vishid = 0.1*np.random.normal(size = (self.numdims, self.numhid)) # v to h weights, 784 x 1000
        hidbiases  = np.zeros((1,self.numhid)) # h to bias weights, 1 x 1000
        visbiases  = np.zeros((1,self.numdims)) # v to bias weights, 1 x 784

        # initialize weights and biases increments
        vishidinc = np.zeros((self.numdims, self.numhid))
        hidbiasinc = np.zeros((1, self.numhid))
        visbiasinc = np.zeros((1, self.numdims))

        # initialize batchposhidprobs
        batchposhidprobs = []

        # initialize error
        error = []

        # iterate over epochs and batches
        for epoch in range(self.maxepoch):
            errsum=0

            for batch in range(len(self.batchdata)):
                # extract one batch of data
                data = self.batchdata[batch]

                # number of examples in this batch
                numcases = data.shape[0]

                # start positive phase
                poshidprobs = (np.dot(data, vishid) + hidbiases)
                batchposhidprobs.append(poshidprobs)
                posprods = np.dot(data.T, poshidprobs)
                poshidact = np.sum(poshidprobs, axis = 0)
                posvisact = np.sum(data, axis = 0)

                # end of positive phase
                poshidstates = poshidprobs + np.random.rand(numcases, self.numhid)

                # start negative phase
                negdata = _logistic((np.dot(poshidstates, vishid.T) + visbiases))
                neghidprobs = np.dot(negdata, vishid) + hidbiases
                negprods = np.dot(negdata.T, neghidprobs)
                neghidact = np.sum(neghidprobs, axis = 0)
                negvisact = np.sum(negdata, axis = 0)

                # end of negative phase
                err = np.sum((data - negdata) ** 2)
                errsum = err + errsum
                error.append(errsum)

                if epoch > 5:
                    momentum = finalmomentum
                else:
                    momentum = initialmomentum

                # update weights and biases
                vishidinc = momentum*vishidinc + epsilonw*( (posprods-negprods)/numcases - weightcost*vishid)
                visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact)
                hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact)
                vishid = vishid + vishidinc
                visbiases = visbiases + visbiasinc
                hidbiases = hidbiases + hidbiasinc

        return (batchposhidprobs, vishid, visbiases, hidbiases, error)