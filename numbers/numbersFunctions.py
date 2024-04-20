import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def euclideanDistance(x, y):
    return np.linalg.norm(x - y) #calculates euclidian distance using norm

def nearestNeighbor(trainingData, trainingLabels, testSample):
    minDistance = float('inf')
    nearestNeighborIdx = None
    
    for i, j in enumerate(trainingData):
        distance = euclideanDistance(j.flatten(), testSample.flatten()) #converts to 1D and looks at euclidian distance between every training sample and the test sample
        if distance < minDistance:
            minDistance = distance
            nearestNeighborIdx = i #assigns the same class label as the closest training sample

    return trainingLabels[nearestNeighborIdx]

def evaluateNearestNeighbor(trainData, trainLabels, testData, testLabels):
    correctPredictions = 0
    totalSamples = len(testData)
    predictedLabels = []
    for i, testSample in enumerate(testData):
        predictedLabel = nearestNeighbor(trainData, trainLabels, testSample)
        predictedLabels.append(predictedLabel)
        if predictedLabel == testLabels[i]:
            correctPredictions += 1
    
    accuracy = correctPredictions / totalSamples
    return accuracy, predictedLabels


