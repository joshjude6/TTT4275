from linalg import euclideanDistance
from timer import Timer # type: ignore

def evaluateNN(trainData, trainLabels, testData, testLabels):
    correctPredictions = 0
    totalSamples = len(testData)
    predictedLabels = []
    correctIndexes = []
    failedIndexes = []
    
    for i, testSample in enumerate(testData):
        predictedLabel = nearestNeighbor(trainData, trainLabels, testSample) #predicts current test sample
        predictedLabels.append(predictedLabel)
        if predictedLabel == testLabels[i]: # -> correct prediction
            correctPredictions += 1
            correctIndexes.append(predictedLabel)
        else:
            failedIndexes.append(predictedLabel)
    accuracy = correctPredictions / totalSamples
    
    return accuracy, predictedLabels, correctIndexes, failedIndexes

def nearestNeighbor(trainingData, trainingLabels, testSample):
    minDistance = float('inf')
    nearestNeighborIndex = None
    
    for i, j in enumerate(trainingData):
        distance = euclideanDistance(j.flatten(), testSample.flatten()) #converts to 1D and looks at euclidian distance between every training sample and the test sample
        if distance < minDistance:
            minDistance = distance
            nearestNeighborIndex = i #assigns the same class label as the closest training sample

    return trainingLabels[nearestNeighborIndex]