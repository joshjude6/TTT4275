import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def euclideanDistance(x, y):
    return np.linalg.norm(x - y) #calculates euclidian distance using norm

def nearestNeighbor(trainingData, trainingLabels, testSample):
    minDistance = float('inf')
    nearestNeighborIndex = None
    
    for i, j in enumerate(trainingData):
        distance = euclideanDistance(j.flatten(), testSample.flatten()) #converts to 1D and looks at euclidian distance between every training sample and the test sample
        if distance < minDistance:
            minDistance = distance
            nearestNeighborIndex = i #assigns the same class label as the closest training sample

    return trainingLabels[nearestNeighborIndex]

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

def KNN(trainingData, trainingLabels, testSample, K):
    distances = []
    for i, j in enumerate(trainingData):
        distance = euclideanDistance(j.flatten(), testSample.flatten()) #converts to 1D and looks at euclidian distance between every training sample and the test sample
        distances.append((distance, i))
    
    KNNindexes = [j for _, j in sorted(distances)[:K]]
    KNNlabels = [trainingLabels[j] for j in KNNindexes] #sorting distances

    labelCounterList = {}
    for x in KNNlabels:
        if x in labelCounterList:
            labelCounterList[x] += 1
        else:
            labelCounterList[x] = 1

    return max(labelCounterList, key=labelCounterList.get) #returns label corresponding to max value

