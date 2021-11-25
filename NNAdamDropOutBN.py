import sys
import numpy as np
import os
import cv2 
from Network import Network
from ActivationType import ActivationType
from LROptimizationType import LROptimizerType
from BatchNormMode import *

def main():
    train = np.empty((1000, 28, 28), dtype='float64')
    trainY = np.zeros((1000,10))
    test = np.empty((10000,28,28), dtype='float64')
    testY = np.zeros((10000, 10))     #Load in the images 
    i = 0
    for filename in os.listdir('D:/Data/Training1000/'):
        y = int(filename[0])
        trainY[i,y] = 1.0 
        train[i] = cv2.imread('D:/Data/Training1000/{0}'.format(filename),0)/255.0 # for color, use 1
        i = i + 1
    i = 0  # read test data 
    for filename in os.listdir('D:/Data/Test10000'):
        testY[i,y] = 1.0
        test[i] = cv2.imread('D:/Data/Test10000/{0}'.format(filename),0)/255.0
        i = i+1
    trainX = train.reshape(train.shape[0], train.shape[1]*train.shape[2])
    testX = test.reshape(test.shape[0], test.shape[1]*test.shape[2])

    numLayers = [50, 10]
    doBatchNorm = True
    NN = Network(trainX, trainY, numLayers,10,1.0,ActivationType.RELU, ActivationType.SIGMOID)
    NN.Train(30,0.1,0.0,20, LROptimizerType.ADAM, doBatchNorm)

    print('done training, starting testing')
    accuracyCount = 0
    for i in range(testY.shape[0]):
        # do forward pass
        a2 = NN.Evaluate(test[i], doBatchNorm, BatchNormMode.TEST)
        #determine index of maximum output value 
        maxindex = a2.armax(axis = 0)
        if (testY[i,maxindex] == 1):
            accuracyCount = accuracyCount + 1
    print("Accuracy count = " + str(accuracyCount/testY.shape[0]*100) + '%')

if __name__ == "__main__":
    sys.exit(int(main() or 0))
