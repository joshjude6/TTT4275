import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time

from datahandler import *
from datetime import datetime
from os import path

# Constants
training_samples = 30
number_of_classes = 3
number_of_features = 4
removed_feature_columns = [] # SL, SW, PL, PW
number_of_features -= len(removed_feature_columns)

training_iterations = 10_000
learning_rate = 0.003

PREFER_PERCENTAGES = True
USE_LAST_N_FOR_TRAINING = False
PLOT_DATA = True
SAVE_PLOTS = False

try:
  dirname, filename = path.split(path.abspath(__file__))
  timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
  temp_path = f'{dirname}/tmp/{timestamp}'
  os.makedirs(temp_path)

  dirname_missing_attr = '' if len(removed_feature_columns) == 0 else f' -- without  {'-'.join(removed_feature_columns)}'
  dirname_ext = f'training {'last' if USE_LAST_N_FOR_TRAINING else 'first'} {training_samples}'
  plot_path = f'{dirname}/plots/{dirname_ext}{dirname_missing_attr}'
  os.makedirs(plot_path)
except FileExistsError: pass

# Fetching data from file
class_0 = get_data(f'{dirname}/data/class_1')
class_1 = get_data(f'{dirname}/data/class_2')
class_2 = get_data(f'{dirname}/data/class_3')

feature_names = class_0.head(0).columns.values
remaining_features = [item for item in feature_names if item not in removed_feature_columns]

# Plot features
if PLOT_DATA:
  for feature in remaining_features:
    column_index = np.where(feature_names == feature)[0]
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0] = sns.histplot(class_0.iloc[:, column_index], ax=axs[0], color='r')
    axs[0].set_ylabel('Antall')
    axs[0].set_title(f'Class 0')
    axs[1] = sns.histplot(class_1.iloc[:, column_index], ax=axs[1], color='g')
    axs[1].set_ylabel('Antall')
    axs[1].set_title(f'Class 1')
    axs[2] = sns.histplot(class_2.iloc[:, column_index], ax=axs[2], color='b')
    axs[2].set_ylabel('Antall')
    axs[2].set_title(f'Class 2')
    plt.suptitle(f'Feature {feature}')
    plt.tight_layout()

    if SAVE_PLOTS:
      fig = fig.get_figure()
      plt.savefig(f'{plot_path}/Feature_{feature}.svg')


class_0 = drop_entry_columns(class_0, column_names=removed_feature_columns)
class_1 = drop_entry_columns(class_1, column_names=removed_feature_columns)
class_2 = drop_entry_columns(class_2, column_names=removed_feature_columns)

if USE_LAST_N_FOR_TRAINING:
  class_0 = shift_last_n_to_beginning(class_0, training_samples)
  class_1 = shift_last_n_to_beginning(class_1, training_samples)
  class_2 = shift_last_n_to_beginning(class_2, training_samples)

# Splitting data sets
class_0_train, class_0_test = separate_data(class_0, training_samples)
class_1_train, class_1_test = separate_data(class_1, training_samples)
class_2_train, class_2_test = separate_data(class_2, training_samples)

# Scatter matrix plot of training data
if PLOT_DATA:
  scatter_matrix = pd.plotting.scatter_matrix(class_0_train, figsize=None, alpha=0.8, c='r')
  pd.plotting.scatter_matrix(class_1_train, alpha=0.8, c='g', ax=scatter_matrix)
  pd.plotting.scatter_matrix(class_2_train, alpha=0.8, c='b', ax=scatter_matrix)
  handles = [plt.plot([],[],color=c, ls="", marker=".", \
              markersize=np.sqrt(10))[0] for c in ['r', 'g', 'b']]
  axis_labels=["Class 0", "Class 1", "Class 2"]
  plt.suptitle('Scatter Matrix of Training Data')
  plt.legend(handles, axis_labels, loc=(1.02,0))
  plt.tight_layout()

  if SAVE_PLOTS:
    plt.savefig(f'{plot_path}/scatter_matrix.svg')

  plt.show()

# Merging data sets
testing_samples = class_0_test.shape[0]
training_data = pd.concat([class_0_train, class_1_train, class_2_train]).values
test_data = pd.concat([class_0_test, class_1_test, class_2_test]).values
print('Data, Training:', training_data.shape)
print('Data, Testing:', test_data.shape)

# Normalising data
max_values = np.array([training_data[:, i].max() for i in range(number_of_features)])
print('Max values:', max_values)
np.savetxt(f'{temp_path}/max_values.txt', max_values)

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
print("\nTraining started...")
start_time = time.time()

for _ in range(training_iterations):
  # Initial condition
  mse_gradient = 0
  mse = 0

  for column in range(number_of_classes * training_samples):
    data = np.array(training_data[column, :])
    data = np.append(data, 1) # concatinate 1x1 matrix to data

    z_k = weight_matrix @ data
    g_k = sigmoid(z_k) # squashing value to 0 or 1
    t_k = training_target_vectors[column, :] # target values

    mse_gradient += compute_mse_gradient(data, t_k, g_k, number_of_features)
    mse += 0.5 * (g_k - t_k).T @ (g_k - t_k)

  mse_values.append(mse)  
  weight_matrix = weight_matrix - learning_rate * mse_gradient

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time:.2f}s")
print("Training finished!\n")

np.set_printoptions(precision=2, suppress=True)
print('Weight matrix:\n', weight_matrix)

if PLOT_DATA:
  plt.plot(mse_values)
  plt.title(f'MSE v. Interation\nLearning rate: {learning_rate}, Iterations: {training_iterations}, Time: {elapsed_time:.3f}s')
  plt.xlabel('Iteration')
  plt.ylabel('MSE')

  if SAVE_PLOTS:
    plt.savefig(f'{plot_path}/MSE_v_Iteration.svg')

  plt.show()

# Confusion matrices. Actual value x predicted value
confusion_matrix_training = np.zeros((number_of_classes, number_of_classes))
confusion_matrix_testing = np.zeros((number_of_classes, number_of_classes))

for i in range(number_of_classes * training_samples):
  data = np.array(training_data[i, :])
  data = np.append(data, 1) # concatinate 1x1 matrix to data
  z_k = weight_matrix @ data
  g_k = sigmoid(z_k) # squashing value to 0 or 1
  t_k = training_target_vectors[i, :]

  if np.argmax(g_k) == np.argmax(t_k):
    confusion_matrix_training[np.argmax(t_k), np.argmax(t_k)] += 1 # hit
  else:
    confusion_matrix_training[np.argmax(t_k), np.argmax(g_k)] += 1 # miss

for i in range(number_of_classes * testing_samples):
  data = np.array(test_data[i, :])
  data = np.append(data, 1) # concatinate 1x1 matrix to data
  z_k = weight_matrix @ data
  g_k = sigmoid(z_k) # squashing value to 0 or 1
  t_k = testing_target_vectors[i, :]

  if np.argmax(g_k) == np.argmax(t_k):
    confusion_matrix_testing[np.argmax(t_k), np.argmax(t_k)] += 1 # hit
  else:
    confusion_matrix_testing[np.argmax(t_k), np.argmax(g_k)] += 1 # miss

print('Confusion matrix, Training:\n', confusion_matrix_training)
print('Confusion matrix, Testing:\n', confusion_matrix_testing)

hits = np.array([np.trace(matrix) for matrix in [confusion_matrix_training, confusion_matrix_testing]])
total_decitions = np.array([np.sum(matrix) for matrix in [confusion_matrix_training, confusion_matrix_testing]])

np.savetxt(f'{temp_path}/total_hits.txt', hits)
np.savetxt(f'{temp_path}/total_decitions.txt', total_decitions)

accuracy = hits / total_decitions
max_accuracy = 100 ** PREFER_PERCENTAGES # 100% or 1.0
accuracy *= max_accuracy

print(f'Accuracy, Training: {accuracy[0]:.2f}{'%' * PREFER_PERCENTAGES}')
print(f'Accuracy, Testing: {accuracy[1]:.2f}{'%' * PREFER_PERCENTAGES}')

print(f'Error rate, Training: {max_accuracy - accuracy[0]:.2f}{'%' * PREFER_PERCENTAGES}')
print(f'Error rate, Testing: {max_accuracy - accuracy[1]:.2f}{'%' * PREFER_PERCENTAGES}')

if PLOT_DATA:
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0] = sns.heatmap(confusion_matrix_training, xticklabels=axis_labels,
                    yticklabels=axis_labels, annot=True, ax=axs[0], annot_kws={"size": 16})
  axs[0].set_title('Training set')
  axs[1] = sns.heatmap(confusion_matrix_testing, xticklabels=axis_labels,
                    yticklabels=axis_labels, annot=True, ax=axs[1], annot_kws={"size": 16})
  axs[1].set_title('Test set')
  plt.suptitle('Confusion matrices')
  plt.tight_layout()

  if SAVE_PLOTS:
    fig = fig.get_figure() # get figure from seaborn
    plt.savefig(f'{plot_path}/Confusion_matrix.svg')

  plt.show()