import math
import numpy as np
from Layer import *
from sklearn.utils import shuffle
from LROptimizationType import LROptimizerType
from BatchNormMode import BatchNormMode

class Network(object):
    def __init__(self, X, Y, numLayers, batchsize, droupOut=1.0,
                 activationF=ActivationType.SIGMOID,lastLayerAF=ActivationType.SIGMOID):
        self.X = X
        self.Y = Y
        self.batchsize = batchsize
        self.numLayers = numLayers
        self.Layers = []
        self.lastLayerAF = lastLayerAF
        for i in range(len(numLayers)):
            if (i == 0):  #first layer
                layer = Layer(numLayers[i], X.shape[1], batchsize,False,droupOut,
                              activationF)
            elif (i == len(numLayers)-1):  #last layer
                    layer = Layer(Y.shape[1], numLayers[i-1],batchsize,True,droupOut,lastLayerAF)
            else: # intermediate layers
                layer = Layer(numLayers[i], numLayers[i-1], batchsize,False,droupOut,activationF)
            self.Layers.append(layer)

    def Evaluate(self, indata, doBatchNorm=False, batchMode=BatchNormMode.TEST):
        # Evaluate all layers
        self.Layers[0].Evaluate(indata, doBatchNorm, batchMode) # first layer
        for i in range(1, len(self.numLayers)):
            self.Layers[i].Evaluate(self.Layers[i-1].a,doBatchNorm,batchMode)
            return self.Layers[len(self.numLayers)-1].a

    def Train(self, epochs, learningRate, lambda1, batchsize=1,
              LROptimization=LROptimizerType.NONE,doBatchNorm=False):
        itnum = 0
        for j in range(epochs):
            error = 0
            self.X, self.Y = shuffle(self.X, self.Y, random_state=0)
            for i in range(0, self.X.shape[0], batchsize):
                # get(X, y) for current minibatch/chunk
                X_train_mini = self.X[i:i + batchsize]
                y_train_mini = self.Y[i:i + batchsize]
                
                self.Evaluate(X_train_mini,doBatchNorm,batchMode=BatchNormMode.TRAIN)
                if (self.lastLayerAF == ActivationType.SOFTMAX):
                    error += -(y_train_mini * np.log(self.Layers[len(self.numLayers)-1].a+0.001)).sum()

                else:
                    error += ((self.Layers[len(self.numLayers)-1].a - y_train_mini) *
                              (self.Layers[len(self.numLayers)-1].a - y_train_mini)).sum()
                    lnum = len(self.numLayers)-1  # last layer number 

                    # compute deltas, grads on all Layers 
                    while (lnum >= 0):
                        if (lnum == len(self.numLayers)-1): #last layer
                            if (self.lastLayerAF == ActivationType.SOFTMAX):
                                self.Layers[lnum].delta = -y_train_mini + self.Layers[lnum].a
                            else:
                                self.Layers[lnum].delta = -y(y_train_mini-self.Layers[lnum].a) * \
                                                            self.Layers[lnum].derivAF
                        else: # intermediate layer
                            self.Layers[lnum].delta= \
                                np.dot(self.Layers[lnum+1].delta,self.Layers[lnum+1].W) * self.Layers[lnum].derivAF
                        if (doBatchNorm == True):
                            self.Layers[lnum].dbeta = np.sum(self.Layers[lnum].delta,axis=0)
                            self.Layers[lnum].dgamma = np.sum(self.Layers[lnum].delta * self.Layers[lnum].Shat,axis=0)
                            self.Layers[lnum].deltabn = (self.Layers[lnum].delta * 
                                                         self.Layers[lnum].gamma)/(batchsize*np.sqrt(self.Layers)
 