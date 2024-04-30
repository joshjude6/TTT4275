from linalg import get_euclidean_distance

def evaluate_nearest_neighbour(training_data, training_labels, test_data, test_labels):
    num_correct_predictions = 0
    num_total_samples = test_data.shape[0]

    predicted_labels = []
    correct_indeces = []
    incorrect_indeces = []

    for i, test_sample in enumerate(test_data):
        label = get_nearest_neighbour(training_data, training_labels, test_sample) #predicts current test sample
        predicted_labels.append(label)
        if label == test_labels[i]: # -> correct prediction
            num_correct_predictions += 1
            correct_indeces.append(label)
        else:
            incorrect_indeces.append(label)
    accuracy = num_correct_predictions / num_total_samples
    
    return accuracy, predicted_labels, correct_indeces, incorrect_indeces

def get_nearest_neighbour(training_data, training_labels, test_sample):
    minimum_distance = float('inf')
    neighbour_index = None
    
    for i, datapoint in enumerate(training_data):
        distance = get_euclidean_distance(datapoint.flatten(), test_sample.flatten()) #converts to 1D and looks at euclidian distance between every training sample and the test sample
        if distance < minimum_distance:
            minimum_distance = distance
            neighbour_index = i #assigns the same class label as the closest training sample

    return training_labels[neighbour_index]