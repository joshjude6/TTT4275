import seaborn as sns
import matplotlib.pyplot as plt

from linalg import euclideanDistance
from sklearn.metrics import confusion_matrix
from timer import Timer # type: ignore


# denne brukes for testing av KNN UTEN clustering
def evaluateKNN(trainData, trainLabels, testData, testLabels, K):
    correctPredictions = 0
    correctIndexes = []
    failedIndexes = [] 
    totalSamples = len(testData)
    predictedLabels = []
    for i, testSample in enumerate(testData):
        predictedLabel = KNN(trainData, trainLabels, testSample, K) #predicts current test sample as seen above
        predictedLabels.append(predictedLabel)
        if predictedLabel == testLabels[i]: # -> correct prediction
            correctPredictions += 1
            correctIndexes.append(i)
        else:
           failedIndexes.append(i)
    
    accuracy = correctPredictions / totalSamples
    return accuracy, predictedLabels, correctIndexes, failedIndexes

# om du vil teste KNN UTEN clustering, bruk denne - NN: sett K til 1
def testKNN(training_data, training_labels, test_data, test_labels, K):
  knn_timer = Timer(f'KNN for full data set, K={K}')
  knn_timer.start()
  accuracyList = np.array([])
  accuracy, predicted_labels, correctIndexes, failedIndexes = evaluateKNN(training_data, training_labels, test_data, test_labels, K)
  accuracyList = np.append(accuracyList, accuracy)
  knn_timer.stop()
  # Plot confusion matrix
  cm = confusion_matrix(test_labels, predicted_labels)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='rocket') #endre farge hvis du vil
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.show()
  print("Total accuracy, with K = {}: {:.2f}%.".format(K, np.average(accuracyList) * 100))


# ordinary KNN, used for non-clustered data
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

# KNN designed to be used with clusered
def knnForKmeans(templates, testSample, K):
    distances = []
    for label, cluster_centers in templates.items():
        for center in cluster_centers:
            distance = euclideanDistance(center, testSample.flatten())
            distances.append((distance, label))
    
    KNNlabels = [label for _, label in sorted(distances)[:K]]

    labelCounterList = {}
    for label in KNNlabels:
        if label in labelCounterList:
            labelCounterList[label] += 1
        else:
            labelCounterList[label] = 1

    return max(labelCounterList, key=labelCounterList.get)

# om du vil teste KNN MED clustering, bruk denne
def evaluateKnnForKmeans(templates, testData, testLabels, K):
    timer = Timer('Evaluating KNN with clustering'); timer.start()
    correctPredictions = 0
    totalSamples = len(testData)
    predictedLabels = []
    for testSample, trueLabel in zip(testData, testLabels):
        predictedLabel = knnForKmeans(templates, testSample, K)
        predictedLabels.append(predictedLabel)
        if predictedLabel == trueLabel:
            correctPredictions += 1
    
    accuracy = correctPredictions / totalSamples
    cm = confusion_matrix(testLabels, predictedLabels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket') #endre farge hvis du vil
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    timer.stop()
    return accuracy, predictedLabels, timeDiff