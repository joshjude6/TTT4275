import pandas as pd
import numpy as np

def get_data(filename='', header=0) -> pd.DataFrame:
    return pd.read_csv(filename, header=header)

def separate_data(data, training_samples=-1) -> tuple[list, list]:
    training_data = data[0:training_samples]
    test_data = data[training_samples:]
    return training_data, test_data

def drop_entry_columns(data, column_names) -> pd.DataFrame:
    data.columns = data.columns.str.strip()
    data = data.drop(columns=column_names)
    return data

def single_value_zero_matrix(shape, position, dtype=float, value=1):
    matrix = np.zeros(shape, dtype=dtype)
    matrix[position] = value
    return matrix

def sigmoid(x):
    return np.array(1 / (1 + np.exp(-x)))

def compute_mse_gradient(data, target, activation_function, number_of_features) -> np.ndarray:
    target_matrix = (activation_function - target) * activation_function * (1 - activation_function)
    target_matrix = target_matrix.reshape(3, 1)
    
    feature_matrix = data
    feature_matrix = feature_matrix.reshape(number_of_features + 1, 1)
    
    gradient = target_matrix @ feature_matrix.T
    return gradient

def shift_last_n_to_beginning(arr, n):
  if n == 0:
    return arr.copy()  # No shift needed, return a copy to avoid modifying original

  if n > arr.shape[0]:
    raise ValueError("n cannot be greater than the array length")

  new_beginning = arr[-n:]
  new_end = arr[:-n]
  return pd.DataFrame(np.concatenate((new_beginning, new_end)))

if __name__ == '__main__':
    data = get_data('data/iris.data')
    print(f'Fetched data of shape ({data.shape})', data)
