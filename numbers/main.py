import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from algos import evaluate_knn
from collections import Counter
from datetime import datetime
from keras.datasets import mnist
from os import path
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from timer import Timer # type: ignore
from numba import jit, types
import time

PLOT_DATA = True
SAVE_PLOTS = False
CHUNK_TRAINING_DATA = False
CHUNK_DATA = True


def euclideanDistance(x, y):
   return np.linalg.norm(x - y)

# nearest neighbour
def nearestNeighbor(trainingData, trainingLabels, testSample):
    minDistance = float('inf')
    nearestNeighborIndex = None
    
    for i, j in enumerate(trainingData):
        distance = euclideanDistance(j.flatten(), testSample.flatten()) #converts to 1D and looks at euclidian distance between every training sample and the test sample
        if distance < minDistance:
            minDistance = distance
            nearestNeighborIndex = i #assigns the same class label as the closest training sample

    return trainingLabels[nearestNeighborIndex]

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
    start = time.time()
    correctPredictions = 0
    totalSamples = len(testData)
    predictedLabels = []
    for testSample, trueLabel in zip(testData, testLabels):
        predictedLabel = knnForKmeans(templates, testSample, K)
        predictedLabels.append(predictedLabel)
        if predictedLabel == trueLabel:
            correctPredictions += 1
    
    accuracy = correctPredictions / totalSamples
    end = time.time()
    timeDiff = end-start
    cm = confusion_matrix(test_labels, predictedLabels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket') #endre farge hvis du vil
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    minutes = int((timeDiff) // 60)
    seconds = int((timeDiff) % 60)
    print(f"Run time: {minutes} minutes and {seconds} seconds.")
    return accuracy, predictedLabels, timeDiff

# selveste clusteringen !! wow
def KMeansClustering(trainingData, trainingLabels, nClusters):
    kmeans = KMeans(n_clusters=nClusters, random_state=0) 
    clusterTemplates = {} 
    for classLabel in np.unique(trainingLabels): 
        classData = trainingData[trainingLabels == classLabel].reshape(-1, (28*28)) 
        kmeans.fit(classData)
        clusterTemplates[classLabel] = kmeans.cluster_centers_ 
    return clusterTemplates

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

def plotSamples(testData, testLabels, predictedLabels, correctIndexes, failedIndexes, N):
    correctSamples = np.random.choice(correctIndexes, min(N//2, len(correctIndexes)), replace=False)
    failedSamples = np.random.choice(failedIndexes, min(N//2, len(failedIndexes)), replace=False)

    plt.figure(figsize=(12, 6))
    for i, j in enumerate(correctSamples):
        plt.subplot(2, N//2, i + 1)
        plt.imshow(testData[j].reshape(28, 28), cmap='rocket')
        plt.title(f'Predicted label: {predictedLabels[j]}\n Actual label: {testLabels[j]}', fontsize=10)
        plt.axis('off')

    for i, j in enumerate(failedSamples):
        plt.subplot(2, N//2, N//2 + i + 1)
        plt.imshow(testData[j].reshape(28, 28), cmap='rocket')
        plt.title(f'Predicted label: {predictedLabels[j]}\n Actual label: {testLabels[j]}', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

  


if __name__ == '__main__':
  timer = Timer()

  try:
    dirname, filename = path.split(path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    plot_path = f'{dirname}/plots'
    os.mkdir(plot_path)
    print('INFO: Writing readable data to temporary file...')
  except FileExistsError:
    print('INFO: Temporary file already exists')

  # Load data
  timer.rename('Data loading')
  timer.start()
  (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
  timer.stop()

  # Split data into chunks
  if CHUNK_DATA:
    timer.rename('Data chunking')
    timer.start()
    samples_per_chunk = 1000
    number_of_training_chunks = training_data.shape[0] // samples_per_chunk  # Integer division for number of chunks
    number_of_label_chunks = training_labels.shape[0] // samples_per_chunk
    training_data = np.array(np.array_split(training_data, number_of_training_chunks, axis=0))
    training_labels = np.array(np.array_split(training_labels, number_of_label_chunks, axis=0))
    print(f'''INFO: Training data split into {number_of_training_chunks} with dimensions {training_data.shape}
INFO: Training labels split into {number_of_label_chunks} with dimensions {training_labels.shape}''')

    number_of_testing_chunks = test_data.shape[0] // samples_per_chunk
    number_of_testing_labels = test_labels.shape[0] // samples_per_chunk
    test_data = np.array(np.array_split(test_data, number_of_testing_chunks, axis=0))
    test_labels = np.array(np.array_split(test_labels, number_of_testing_labels, axis=0))
    print(f'''INFO: Test data split into {number_of_testing_chunks} with dimensions {test_data.shape}
INFO: Test labels split into {number_of_testing_labels} with dimensions {test_labels.shape}''')
    timer.stop()

  print(f'''Training data: {training_data.shape}
Training labels: {training_labels.shape}
Test data: {test_data.shape}
Test labels: {test_labels.shape}
''')

  # Normalize data
  training_data = training_data / 255
  test_data = test_data / 255
  # One hot encode labels
  
  one_hot_encode = lambda labels: np.eye(max(labels + 1))[labels]
  encoded_training_labels = np.array([one_hot_encode(label_chunk) for label_chunk in training_labels])
  encoded_test_labels = np.array([one_hot_encode(label_chunk) for label_chunk in test_labels])
  
for i in range(8):
    evaluateNN(training_data[i], training_labels[i], test_data[i], test_labels[i])