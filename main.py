import numpy as np
import os
import pandas as pd

from datahandler import get_data, drop_entry_columns, separate_data

try:
  os.mkdir('tmp')
except FileExistsError: pass

# pushed

# Fetching data from file
class_0 = get_data('data/class_1')
class_1 = get_data('data/class_2')
class_2 = get_data('data/class_3')

training_samples = 30
number_of_classes = 3
number_of_features = 4
removed_feature_columns = [] # SL, SW, PL, PW
number_of_features -= len(removed_feature_columns)

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
print('Training data:', training_data.shape)
print('Test data:', test_data.shape)

# Computing max values per column
max_values = np.array([training_data[:, i].max() for i in range(number_of_features)])
print('Max values:', max_values)
np.savetxt('tmp/max_values.txt', max_values)

training_data_normalised = training_data / max_values
np.savetxt('tmp/normalised_training_data.txt', training_data_normalised)