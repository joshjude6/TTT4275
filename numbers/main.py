import matplotlib.pyplot as plt
import numpy as np
import os
from keras.datasets import mnist
import seaborn as sns

from algos import evaluate_knn
from datetime import datetime
from keras.datasets import mnist
from os import path
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from timer import Timer # type: ignore

PLOT_DATA = False
SAVE_PLOTS = False
CHUNK_TRAINING_DATA = True

def testKNN(training_data, training_labels, test_data, test_labels, K):
  accuracyList = np.array([])
  if 3 == training_data.ndim:
    knn_timer = Timer(f'KNN for full data set, K={K}')
    knn_timer.start()
    accuracy, predicted_labels = evaluate_knn(training_data, training_labels, test_data, test_labels, K)
    knn_timer.stop()
    print(f'Accuracy: {accuracy}')
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
    knn_timer = Timer(f'KNN on {test_data.shape[0]} chunks, K={K}')
    knn_timer.start()
    for i in range(test_data.shape[0]):
      accuracy, predicted_labels = evaluate_knn(training_data[i], training_labels[i], test_data[i], test_labels[i], K)
      print(f'Chunk #{i + 1} Accuracy: {accuracy}')
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
    knn_timer.stop()

  if PLOT_DATA:
    plt.show()
    
  print(f"Total accuracy for set of chunks, with K = {K}: {np.average(accuracyList)*100:.2f}%.")


if __name__ == '__main__':
  main_process_timer = Timer('Main process')
  main_process_timer.start()
  
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

  print('HERE', type(timer), timer)
  
  print(f'''Training data: {training_data.shape}
Training labels: {training_labels.shape}
Test data: {test_data.shape}
Test labels: {test_labels.shape}
''')

  # Split data into chunks
  if CHUNK_TRAINING_DATA:
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

  print('Encoding labels...')
  timer.rename('Target vector encoding')
  timer.start()
  if 1 < training_labels.ndim:
    encoded_training_labels = np.array([one_hot_encode(label_chunk) for label_chunk in training_labels])
    encoded_test_labels = np.array([one_hot_encode(label_chunk) for label_chunk in test_labels])
  else:
    encoded_training_labels = np.array(one_hot_encode(training_labels))
    encoded_test_labels = np.array(one_hot_encode(test_labels))
  timer.stop()
     

  print(f'''Encoded Training Labels: {encoded_training_labels.shape}
Encoded Test Labels {encoded_test_labels.shape}
''')
  
  k = 7
  testKNN(training_data, training_labels, test_data, test_labels, k)
  main_process_timer.stop()

      


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