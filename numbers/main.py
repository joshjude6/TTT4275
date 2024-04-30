import matplotlib.pyplot as plt
import numpy as np
import os

from algos.clustering import KMeansClustering
from algos.k_nearest_neighbour import KNN, knnForKmeans, evaluateKnnForKmeans, evaluateKNN, testKNN
from algos.nearest_neighbour import evaluate_nearest_neighbour, get_nearest_neighbour
from collections import Counter
from datetime import datetime
from keras.datasets import mnist # type: ignore
from os import path
from timer import Timer # type: ignore

# Runtime configurations
PLOT_DATA = True
SAVE_PLOTS = False

ENABLE_CHUNKING = True
samples_per_chunk = 1000

def plot_samples(test_data, test_labels, predicted_labels, correct_indeces, failed_indeces, N) -> None:
    correct_samples = np.random.choice(correct_indeces, min(N//2, len(correct_indeces)), replace=False)
    failed_samples = np.random.choice(failed_indeces, min(N//2, len(failed_indeces)), replace=False)

    plt.figure(figsize=(12, 6))
    for i, j in enumerate(correct_samples):
        plt.subplot(2, N//2, i + 1)
        plt.imshow(test_data[j].reshape(28, 28), cmap='rocket')
        plt.title(f'Predicted label: {predicted_labels[j]}\n Actual label: {test_labels[j]}', fontsize=10)
        plt.axis('off')

    for i, j in enumerate(failed_samples):
        plt.subplot(2, N//2, N//2 + i + 1)
        plt.imshow(test_data[j].reshape(28, 28), cmap='rocket')
        plt.title(f'Predicted label: {predicted_labels[j]}\n Actual label: {test_labels[j]}', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main program
if __name__ == '__main__':
  timer = Timer()

  # Create directory for plots
  try:
    dirname, filename = path.split(path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    plot_path = f'{dirname}/plots'
    os.makedirs(plot_path)
    print('INFO: Writing readable data to temporary file...')
  except FileExistsError:
    print('INFO: Temporary file already exists')

  # Load data
  timer.rename('Data loading')
  timer.start()
  (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
  timer.stop()

  # Split data into chunks
  if ENABLE_CHUNKING:
    timer.rename('Data chunking')
    timer.start()

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
  
  timer.rename('Evaluate NN for multiple chunks')
  timer.start()
  for i in range(8):
    evaluate_nearest_neighbour(training_data[i], training_labels[i], test_data[i], test_labels[i])
  timer.stop()