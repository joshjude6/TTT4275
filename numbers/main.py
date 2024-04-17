import matplotlib.pyplot as plt
import numpy as np
import os
import struct

from os import path

SHOW_PLOT = False

def read_bin(path):
  with open(path, 'rb') as file:
    _, size = struct.unpack(">II", file.read(8))
    nrows, ncols = struct.unpack(">II", file.read(8))
    data = np.fromfile(file, dtype='uint8').reshape((size, nrows, ncols))

  return data, (nrows, ncols)

if __name__ == '__main__':
  dirname, filename = path.split(path.abspath(__file__))
  training_data, _ = read_bin(f'{dirname}/data/train_images.bin')
  print('Training data:', training_data.shape)
  # training_labels, _ = read_bin(f'{dirname}/data/train_labels.bin')
  # print('Training labels:', training_labels.shape)
  
  # test_data, _ = read_bin(f'{dirname}/data/test_images.bin')
  # print('Test data:', test_data.shape)
  # test_labels, _ = read_bin(f'{dirname}/data/test_labels.bin')
  # print('Test labels:', test_labels.shape)

  try:
    os.mkdir(f'{dirname}/tmp')
    print('INFO: Writing readable data to temporary file...')
    np.savetxt(f'{dirname}/tmp/data.txt', training_data)
    print('Data written to file...')
  except FileExistsError:
    print('INFO: Directory already exists')

  samples_per_chunk = 1000
  number_of_chunks = training_data.shape[0] // samples_per_chunk  # Integer division for number of chunks
  chunked_data = np.array_split(training_data, number_of_chunks, axis=0)
  print(f'INFO: Training data split into {number_of_chunks} chunks of size {samples_per_chunk}')

  if SHOW_PLOT:
    # Plot first 9 samples from chunk 0
    for i in range(9):
      first_samples = chunked_data[0][:9]
      plt.subplot(330 + 1 + i)
      plt.imshow(first_samples[i], cmap=plt.get_cmap('gray'))
      plt.suptitle('First 9 samples from chunk 0')
      plt.tight_layout()
    plt.show()