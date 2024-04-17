import matplotlib.pyplot as plt
import numpy as np
import os
import struct

from keras.datasets import mnist
from os import path

SHOW_PLOTS = False
SAVE_PLOTS = False

def read_bin(path, dtype='uint8', reshape=False):
  with open(path, 'rb') as file:
    _, size = struct.unpack(">II", file.read(8))
    nrows, ncols = struct.unpack(">II", file.read(8))
    data = np.fromfile(file, dtype=dtype)

    if reshape:
      data = data.reshape((size, nrows, ncols))

  return data

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
INFO: Training labels split into {number_of_label_chunks} chunks of size {len(chunked_training_labels[0])}
''')

  number_of_testing_chunks = test_data.shape[0] // samples_per_chunk
  number_of_testing_labels = test_labels.shape[0] // samples_per_chunk
  chunked_test_data = np.array(np.array_split(test_data, number_of_testing_chunks, axis=0))
  chunked_test_labels = np.array(np.array_split(test_labels, number_of_testing_labels, axis=0))
  print(f'''INFO: Test data split into {number_of_testing_chunks} chunks of size {chunked_test_data.shape[1]}
INFO: Test labels split into {number_of_testing_labels} chunks of size {len(chunked_test_labels[0])}
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