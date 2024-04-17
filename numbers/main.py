import matplotlib.pyplot as plt
import numpy as np
import os
import struct

from os import path

SHOW_PLOT = False

def read_bin(path, reshape=False):
  with open(path, 'rb') as file:
    _, size = struct.unpack(">II", file.read(8))
    nrows, ncols = struct.unpack(">II", file.read(8))
    data = np.fromfile(file, dtype='uint8')

    if reshape:
      data = data.reshape((size, nrows, ncols))

  return data

if __name__ == '__main__':
  dirname, filename = path.split(path.abspath(__file__))
  training_data = read_bin(f'{dirname}/data/train_images.bin', reshape=True)
  print('Training data:', training_data.shape)
  training_labels = read_bin(f'{dirname}/data/train_labels.bin')
  print('Training labels:', training_labels.shape)
  
  test_data = read_bin(f'{dirname}/data/test_images.bin', reshape=True)
  print('Test data:', test_data.shape)
  test_labels = read_bin(f'{dirname}/data/test_labels.bin')
  print('Test labels:', test_labels.shape)

  try:
    os.mkdir(f'{dirname}/tmp')
    print('INFO: Writing readable data to temporary file...')
    np.savetxt(f'{dirname}/tmp/data.txt', training_data)
    print('Data written to file...')
  except FileExistsError:
    print('INFO: Directory already exists')

  # Split data into chunks
  samples_per_chunk = 1000
  number_of_training_chunks = training_data.shape[0] // samples_per_chunk  # Integer division for number of chunks
  number_of_label_chunks = training_labels.shape[0] // samples_per_chunk
  chunked_data = np.array_split(training_data, number_of_training_chunks, axis=0)
  chuncked_labels = np.array_split(training_labels, number_of_label_chunks, axis=0)
  print(f'INFO: Training data split into {number_of_training_chunks} chunks of size {samples_per_chunk}')
  print(f'INFO: Training labels split into {number_of_label_chunks} chunks of size {samples_per_chunk}')

  number_of_testing_chunks = test_data.shape[0] // samples_per_chunk
  number_of_testing_labels = test_labels.shape[0] // samples_per_chunk
  chunked_test_data = np.array_split(test_data, number_of_testing_chunks, axis=0)
  chunked_test_labels = np.array_split(test_labels, number_of_testing_labels, axis=0)
  print(f'INFO: Test data split into {number_of_testing_chunks} chunks of size {samples_per_chunk}')
  print(f'INFO: Test labels split into {number_of_testing_labels} chunks of size {samples_per_chunk}')

  if SHOW_PLOT:
    # Plot first 9 samples from chunk 0
    for data_set in [chunked_data, chunked_test_data]:
      for i in range(9):
        first_samples = data_set[0][:9]
        plt.subplot(330 + 1 + i)
        plt.imshow(first_samples[i], cmap=plt.get_cmap('gray'))
        plt.suptitle('First 9 samples from chunk 0')
        plt.tight_layout()
      plt.show()