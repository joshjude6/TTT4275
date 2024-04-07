import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

from datahandler import *

try:
  os.mkdir('tmp')
except FileExistsError: pass

# Constants
training_samples = 30
number_of_classes = 3
number_of_features = 4
removed_feature_columns = [] # SL, SW, PL, PW
number_of_features -= len(removed_feature_columns)

training_iterations = 1000
learning_rate = 0.005
PLOT_DATA = False

# Fetching data from file
class_0 = get_data('data/class_1')
class_1 = get_data('data/class_2')
class_2 = get_data('data/class_3')

class_0 = drop_entry_columns(class_0, column_names=removed_feature_columns)
class_1 = drop_entry_columns(class_1, column_names=removed_feature_columns)
class_2 = drop_entry_columns(class_2, column_names=removed_feature_columns)

# Splitting data sets
class_0_train, class_0_test = separate_data(class_0, training_samples)
class_1_train, class_1_test = separate_data(class_1, training_samples)
class_2_train, class_2_test = separate_data(class_2, training_samples)

# Merging data sets
training_data = pd.concat([class_0_train, class_1_train, class_2_train]).values
test_data = pd.concat([class_0_test, class_1_test, class_2_test]).values
testing_samples = test_data.shape[0]
print('Data, Training:', training_data.shape)
print('Data, Testing:', test_data.shape)

# Normalising data
max_values = np.array([training_data[:, i].max() for i in range(number_of_features)])
print('Max values:', max_values)
np.savetxt('tmp/max_values.txt', max_values)

training_data_normalised = training_data / max_values
np.savetxt('tmp/normalised_training_data.txt', training_data_normalised)

# Setting up target vectors and weight matrix
target_vectors = [single_value_zero_matrix(shape=number_of_classes, position=i, value=1) for i in range(number_of_classes)]
training_target_vectors = np.vstack([np.tile(vec, (training_samples, 1)) for vec in target_vectors])
testing_target_vectors = np.vstack([np.tile(vec, (testing_samples, 1)) for vec in target_vectors])
print('Target vector, Training:', training_target_vectors.shape)
print('Target vector, Testing:', testing_target_vectors.shape)

weight_matrix = np.zeros((number_of_classes, number_of_features + 1))
print('Weight matrix:', weight_matrix.shape)

# Training
mse_values = []
print("\nStarting training")
start_time = time.time()

for _ in range(training_iterations):
  # Initial condition
  mse_gradient = 0
  mse = 0

  for column in range(number_of_classes * training_samples):
    row_data = np.array(training_data[column, :])
    row_data = np.append(row_data, 1)
    z_k = weight_matrix @ row_data
    g_k = sigmoid(z_k)
    t_k = training_target_vectors[column, :]
    mse_gradient += compute_mse_gradient(row_data, t_k, g_k, number_of_features)
    mse += 0.5 * (g_k - t_k).T @ (g_k - t_k)

  mse_values.append(mse)  
  weight_matrix = weight_matrix - learning_rate * mse_gradient

end_time = time.time()
elapsed_time = round(end_time - start_time, 2)
print("Training time: ", elapsed_time, "s")
print("Training done\n")

np.set_printoptions(precision=2, suppress=True)
print('Weight matrix:', weight_matrix)

if PLOT_DATA:
  plt.plot(mse_values)
  plt.title(f'MSE v. Interation\nLearning rate: {learning_rate}, Iterations: {training_iterations}, Time: {elapsed_time}s')
  plt.xlabel('Iteration')
  plt.ylabel('MSE')
  plt.show()