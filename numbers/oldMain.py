import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import time

from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from os import path

SHOW_PLOTS = False
SAVE_PLOTS = False
CHUNK_TRAINING_DATA = True

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
  if CHUNK_TRAINING_DATA:
    samples_per_chunk = 1000
    number_of_training_chunks = training_data.shape[0] // samples_per_chunk  # Integer division for number of chunks
    number_of_label_chunks = training_labels.shape[0] // samples_per_chunk
    training_data = np.array(np.array_split(training_data, number_of_training_chunks, axis=0))
    training_labels = np.array(np.array_split(training_labels, number_of_label_chunks, axis=0))
    print(f'''INFO: Training data split into {number_of_training_chunks} chunks of size {training_data.shape[1]}
INFO: Training labels split into {number_of_label_chunks} chunks of size {training_labels.shape[1]}''')

    number_of_testing_chunks = test_data.shape[0] // samples_per_chunk
    number_of_testing_labels = test_labels.shape[0] // samples_per_chunk
    test_data = np.array(np.array_split(test_data, number_of_testing_chunks, axis=0))
    test_labels = np.array(np.array_split(test_labels, number_of_testing_labels, axis=0))
    print(f'''INFO: Test data split into {number_of_testing_chunks} chunks of size {test_data.shape[1]}
INFO: Test labels split into {number_of_testing_labels} chunks of size {test_labels.shape[1]}
''')
  
  print(f'''Training data: {training_data.shape}
Training labels: {training_labels.shape}
Test data: {test_data.shape}
Test labels: {test_labels.shape}
''')

  if SHOW_PLOTS:
    # Plot first 9 samples from chunk 0
    for data_set in [training_data, test_data]:
      for i in range(9):
        first_samples = data_set[0][:9]
        plt.subplot(330 + 1 + i)
        plt.imshow(first_samples[i], cmap=plt.get_cmap('gray'))
        plt.suptitle('First 9 samples from chunk 0')
        plt.tight_layout()
      plt.show()

  # Normalize data
  training_data = training_data / 255
  test_data = test_data / 255

  # One hot encode labels
  one_hot_encode = lambda labels: np.eye(max(labels + 1))[labels]
  encoded_training_labels = np.array([one_hot_encode(label_chunk) for label_chunk in training_labels])
  encoded_test_labels = np.array([one_hot_encode(label_chunk) for label_chunk in test_labels])
  print(f'''Encoded Training Labels: {encoded_training_labels.shape}
Encoded Test Labels {encoded_test_labels.shape}
''')
  
  # Evaluate nearest neighbor classifier
  print('Evaluating for each test data chunk...')
  start_time = time.time()
  for i in range(test_data.shape[0]):
    chunk_timer_start = time.time()
    accuracy, predicted_labels = evaluateNearestNeighbor(training_data[i], training_labels[i], test_data[i], test_labels[i])
    print(f'Chunk #{i + 1} Accuracy: {accuracy} | Time elapsed: {time.time() - chunk_timer_start:.2f}s')

    # Plot confusion matrix
    cm = confusion_matrix(test_labels[i], predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

  print('Finished!')
  print(f"Total time elapsed: {time.time() - start_time:.2f}s")

  if SHOW_PLOTS:
    plt.show()