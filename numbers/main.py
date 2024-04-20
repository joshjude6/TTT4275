import matplotlib.pyplot as plt
import numpy as np
import os
import numbersFunctions
from keras.datasets import mnist
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from os import path

SHOW_PLOTS = True
SAVE_PLOTS = False

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

def evaluateNearestNeighbor(trainData, trainLabels, testData, testLabels):
    correctPredictions = 0
    totalSamples = len(testData)
    predictedLabels = []
    for i, testSample in enumerate(testData):
        predictedLabel = KNN(trainData, trainLabels, testSample, 7) #predicts current test sample as seen above
        predictedLabels.append(predictedLabel)
        if predictedLabel == testLabels[i]: # -> correct prediction
            correctPredictions += 1
    
    accuracy = correctPredictions / totalSamples
    return accuracy, predictedLabels

if __name__ == '__main__':
  if SAVE_PLOTS:
    try:
      dirname, filename = path.split(path.abspath(__file__))
      os.mkdir(f'{dirname}/plots')
      print('INFO: Writing readable data to temporary file...')
    except FileExistsError:
      print('INFO: Temporary file already exists')

  # Load data
  (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
  
  print(f'''Training data: {training_data.shape}
Training labels: {training_labels.shape}
Test data: {test_data.shape}
Test labels: {test_labels.shape}
''')

  # Split data into chunks
  samples_per_chunk = 1000
  number_of_training_chunks = training_data.shape[0] // samples_per_chunk  # Integer division for number of chunks
  number_of_label_chunks = training_labels.shape[0] // samples_per_chunk
  chunked_training_data = np.array(np.array_split(training_data, number_of_training_chunks, axis=0))
  chunked_training_labels = np.array(np.array_split(training_labels, number_of_label_chunks, axis=0))
  print(f'''INFO: Training data split into {number_of_training_chunks} chunks of size {chunked_training_data.shape[1]}
INFO: Training labels split into {number_of_label_chunks} chunks of size {chunked_training_labels.shape[1]}''')

  number_of_testing_chunks = test_data.shape[0] // samples_per_chunk
  number_of_testing_labels = test_labels.shape[0] // samples_per_chunk
  chunked_test_data = np.array(np.array_split(test_data, number_of_testing_chunks, axis=0))
  chunked_test_labels = np.array(np.array_split(test_labels, number_of_testing_labels, axis=0))
  print(f'''INFO: Test data split into {number_of_testing_chunks} chunks of size {chunked_test_data.shape[1]}
INFO: Test labels split into {number_of_testing_labels} chunks of size {chunked_test_labels.shape[1]}
''')
  
  print(f'''Training data: {chunked_training_data.shape}
Training labels: {chunked_training_labels.shape}
Test data: {chunked_test_data.shape}
Test labels: {chunked_test_labels.shape}
''')

  if SHOW_PLOTS:
    # Plot first 9 samples from chunk 0
    for data_set in [chunked_training_data, chunked_test_data]:
      for i in range(9):
        first_samples = data_set[0][:9]
        plt.subplot(330 + 1 + i)
        plt.imshow(first_samples[i], cmap=plt.get_cmap('gray'))
        plt.suptitle('First 9 samples from chunk 0')
        plt.tight_layout()
      plt.show()

  # Normalize data
  chunked_training_data = chunked_training_data / 255
  chunked_test_data = chunked_test_data / 255

  # One hot encode labels
  one_hot_encode = lambda labels: np.eye(max(labels + 1))[labels]
  encoded_training_labels = np.array([one_hot_encode(label_chunk) for label_chunk in chunked_training_labels])
  encoded_test_labels = np.array([one_hot_encode(label_chunk) for label_chunk in chunked_test_labels])
  print(f'''Encoded Training Labels: {encoded_training_labels.shape}
Encoded Test Labels {encoded_test_labels.shape}
''')
  
  # Evaluate nearest neighbor classifier
  for i in range(9):
    accuracy, predicted_labels = evaluateNearestNeighbor(chunked_training_data[i], chunked_training_labels[i], chunked_test_data[i], chunked_test_labels[i])
    print("Accuracy:", accuracy)

    # Plot confusion matrix
    cm = confusion_matrix(chunked_test_labels[i], predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()