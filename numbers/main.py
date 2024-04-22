import matplotlib.pyplot as plt
import numpy as np
import os
import time
from keras.datasets import mnist
import seaborn as sns


from datetime import datetime
from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from os import path

PLOT_DATA = True
SAVE_PLOTS = True
CHUNK_TRAINING_DATA = False

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

def evaluateKNN(trainData, trainLabels, testData, testLabels, K):
    correctPredictions = 0
    totalSamples = len(testData)
    predictedLabels = []
    for i, testSample in enumerate(testData):
        predictedLabel = KNN(trainData, trainLabels, testSample, K) #predicts current test sample as seen above
        predictedLabels.append(predictedLabel)
        if predictedLabel == testLabels[i]: # -> correct prediction
            correctPredictions += 1
    
    accuracy = correctPredictions / totalSamples
    return accuracy, predictedLabels

def testKNN(training_data, training_labels, test_data, test_labels, K):
  print(f"Testing with K = {K}")
  start = time.time()
  accuracyList = np.array([])
  if 3 == training_data.ndim:
    print('Running KNN on full data set...')
    chunk_timer_start = time.time()
    accuracy, predicted_labels = evaluateKNN(training_data, training_labels, test_data, test_labels, K)
    print(f'Accuracy: {accuracy} | Time elapsed: {(time.time() - chunk_timer_start) / 60:.2f}min')
    accuracyList = np.append(accuracyList, accuracy)
    # Plot confusion matrix
    if PLOT_DATA:
      cm = confusion_matrix(test_labels, predicted_labels)
      plt.figure(figsize=(10, 8))
      sns.heatmap(cm, annot=True, fmt='d', cmap='crest') #endre farge hvis du vil
      plt.title(f'Confusion Matrix, Full Data Set\nK={K}')
      plt.xlabel('Predicted')
      plt.ylabel('Actual')
      plt.tight_layout()

      if SAVE_PLOTS:
        plt.savefig(f'{plot_path}/confusion_matrix{'__k_' + str(K)}.svg')

  else:
    print(f'Running KNN on {test_data.shape[0]} chunks...')
    for i in range(test_data.shape[0]):
      chunk_timer_start = time.time()
      accuracy, predicted_labels = evaluateKNN(training_data[i], training_labels[i], test_data[i], test_labels[i], K)
      print(f'Chunk #{i + 1} Accuracy: {accuracy} | Time elapsed: {time.time() - chunk_timer_start:.2f}s')
      accuracyList = np.append(accuracyList, accuracy)
      # Plot confusion matrix
      if PLOT_DATA:
        cm = confusion_matrix(test_labels[i], predicted_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix, Chunk size {test_labels.shape[1]}\nChunk ' + str(i) + ', K=' + str(K))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        if SAVE_PLOTS:
          plt.savefig(f'{plot_path}/confusion_matrix{'__k_' + str(K)}{'__chunk_' + str(i) if CHUNK_TRAINING_DATA else ''}.svg')

  if PLOT_DATA:
    plt.show()
    
  print(f"Total accuracy for set of chunks, with K = {K}: {np.average(accuracyList)*100:.2f}%.")
  end = time.time()
  print(f"Run time: {((end-start)/60):.2f} minutes.")


if __name__ == '__main__':
  try:
    dirname, filename = path.split(path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    plot_path = f'{dirname}/plots'
    os.mkdir(plot_path)
    print('INFO: Writing readable data to temporary file...')
  except FileExistsError:
    print('INFO: Temporary file already exists')

  # Load data
  print('Loading data...')
  timer_start = time.time()
  (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
  print(f'Loading done! | Time elapsed: {time.time() - timer_start:.2f}s')
  
  print(f'''Training data: {training_data.shape}
Training labels: {training_labels.shape}
Test data: {test_data.shape}
Test labels: {test_labels.shape}
''')

  # Split data into chunks
  if CHUNK_TRAINING_DATA:
    print('Chunking data...')
    timer_start = time.time()
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
    print(f'Chunking done! | Time elapsed: {time.time() - timer_start:.2f}s\n')
  
  print(f'''Training data: {training_data.shape}
Training labels: {training_labels.shape}
Test data: {test_data.shape}
Test labels: {test_labels.shape}
''')

  # Plot first 9 samples from chunk 0
  # if PLOT_DATA:
  #   for data_set in [training_data, test_data]:
  #     for i in range(9):
  #       first_samples = data_set[0][:9]
  #       plt.subplot(330 + 1 + i)
  #       plt.imshow(first_samples[i], cmap=plt.get_cmap('gray'))
  #       plt.suptitle('First 9 samples from chunk 0')
  #       plt.tight_layout()
          
  #     plt.show()

  # Normalize data
  training_data = training_data / 255
  test_data = test_data / 255

  # One hot encode labels
  one_hot_encode = lambda labels: np.eye(max(labels + 1))[labels]

  print('Encoding labels...')
  timer_start = time.time()
  if 1 < training_labels.ndim:
    encoded_training_labels = np.array([one_hot_encode(label_chunk) for label_chunk in training_labels])
    encoded_test_labels = np.array([one_hot_encode(label_chunk) for label_chunk in test_labels])
  else:
    encoded_training_labels = np.array(one_hot_encode(training_labels))
    encoded_test_labels = np.array(one_hot_encode(test_labels))
  print(f'Encoding done! | Time elapsed: {time.time() - timer_start:.2f}s')
     

  print(f'''Encoded Training Labels: {encoded_training_labels.shape}
Encoded Test Labels {encoded_test_labels.shape}
''')
  

      


  ''' Kode som ikke funker
  def clusteringData(training_data, training_labels, numClusters):
    flatData = training_data.reshape(training_data.shape[0], -1) 
    flatLabels = training_labels.reshape(training_data.shape[0], -1) 
    kmeans = KMeans(numClusters, random_state=0)
    kmeans.fit(flatData) #fitting kmeans to flattened training data
    clusterLabels = kmeans.labels_

    clusteredData = []
    clusteredLabels = [] 

    for i in range(numClusters):
        data = flatData[clusteredData == i]
        labels = flatLabels[clusteredLabels == i]
        clusteredData.append(data)
        clusteredLabels.append(labels)
    
    return clusteredData, clusteredLabels

  clustered_training_data, clustered_training_labels = clusteringData(chunked_training_data, chunked_training_labels, 64)

  # Test KNN with clustered data
  print(f"Testing with K = 5, using 64 clusters")
  testKNN(clustered_training_data, clustered_training_labels, chunked_test_data, chunked_test_labels, 5)
  ''' 

  # Testing av KNN og NN uten clustering
  
  k = 7
  timer_start = time.time()
  testKNN(training_data, training_labels, test_data, test_labels, k)
  print('Finished!')
  total_time = time.time() - timer_start
  show_minutes = False
  if 120 < total_time:
    total_time /= 60
    show_minutes = True
  print(f"Total time elapsed: {total_time:.2f}{'min' if show_minutes else 's'}")