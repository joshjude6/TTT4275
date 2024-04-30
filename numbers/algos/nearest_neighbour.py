from linalg import euclideanDistance

def nearestNeighbor(trainingData, trainingLabels, testSample):
    minDistance = float('inf')
    nearestNeighborIndex = None
    
    for i, j in enumerate(trainingData):
        distance = euclideanDistance(j.flatten(), testSample.flatten()) #converts to 1D and looks at euclidian distance between every training sample and the test sample
        if distance < minDistance:
            minDistance = distance
            nearestNeighborIndex = i #assigns the same class label as the closest training sample

    return trainingLabels[nearestNeighborIndex]