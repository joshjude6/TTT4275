import numpy as np

from linalg import euclidean_distance
from sklearn.cluster import KMeans

def nearest_neighbour(training_data, training_labels, test_sample):
    least_distance_discovered = float('inf')
    nearest_neighbour_index = None
    
    for index, sample in enumerate(training_data):
        distance = euclidean_distance(sample.flatten(), test_sample.flatten()) #converts to 1D and looks at euclidian distance between every training sample and the test sample
        if distance < least_distance_discovered:
            least_distance_discovered = distance
            nearest_neighbour_index = index #assigns the same class label as the closest training sample

    return training_labels[nearest_neighbour_index]

def k_nearest_neighbours(training_data, training_labels, test_sample, k=1):
    distances = np.array([])
    # distances = []
    for index, sample in enumerate(training_data):
        distance = euclidean_distance(sample.flatten(), test_sample.flatten()) #converts to 1D and looks at euclidian distance between every training sample and the test sample
        distances = np.append(distances, (distance, index))
        # distances.append((distance, index))
    
    if len(distances.shape) == 1:
      distances = distances.reshape(-1, 2)

    # sort by distance and get nearset k indeces
    sorted_distances = distances[distances[:, 0].argsort()]
    nearest_neighbours = sorted_distances[:k]
    neighbour_indeces = nearest_neighbours[:, 1].astype(int)
    
    knn_labels = training_labels[neighbour_indeces]

    label_counter = {}
    for x in knn_labels:
        if x in label_counter:
            label_counter[x] += 1
        else:
            label_counter[x] = 1

    return max(label_counter, key=label_counter.get) #returns label corresponding to max value

def evaluate_knn(train_data, train_labels, test_data, test_labels, k=1):
    correct_prediction_counter = 0
    total_samples = test_data.shape[0]
    predicted_labels = np.array([])
    for index, test_sample in enumerate(test_data):
        predicted_label = k_nearest_neighbours(train_data, train_labels, test_sample, k) #predicts current test sample as seen above
        predicted_labels = np.append(predicted_labels, predicted_label)
        if predicted_label == test_labels[index]: # -> correct prediction
            correct_prediction_counter += 1
    
    accuracy = correct_prediction_counter / total_samples
    return accuracy, predicted_labels

