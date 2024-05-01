import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from linalg import get_euclidean_distance
from sklearn.metrics import confusion_matrix
from timer import Timer  # type: ignore


# denne brukes for testing av KNN UTEN clustering
def evaluate_knn(training_data, training_labels, test_data, test_labels, K):
  num_correct_predictions = 0
  num_total_samples = test_data.shape[0]

  predicted_labels = []
  correct_indeces = []
  failed_indeces = []

  for i, test_sample in enumerate(test_data):
    label = get_k_nearest_neighbour(
      training_data, training_labels, test_sample, K
    )  # predicts current test sample as seen above
    predicted_labels.append(label)

    if label == test_labels[i]:  # -> correct prediction
      num_correct_predictions += 1
      correct_indeces.append(i)
    else:
      failed_indeces.append(i)

  accuracy = num_correct_predictions / num_total_samples
  return accuracy, predicted_labels, correct_indeces, failed_indeces


# om du vil teste KNN UTEN clustering, bruk denne - NN: sett K til 1
def test_knn(training_data, training_labels, test_data, test_labels, K=1):
  knn_timer = Timer(f"KNN for full data set, K={K}")
  knn_timer.start()
  accuracyList = np.array([])
  accuracy, predicted_labels, correctIndexes, failedIndexes = evaluate_knn(
    training_data, training_labels, test_data, test_labels, K
  )
  accuracyList = np.append(accuracyList, accuracy)
  knn_timer.stop()

  # Plot confusion matrix
  cm = confusion_matrix(test_labels, predicted_labels)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt="d", cmap="rocket")  # endre farge hvis du vil
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.show()
  print(
    "Total accuracy, with K = {}: {:.2f}%.".format(K, np.average(accuracyList) * 100)
  )


# ordinary KNN, used for non-clustered data
def get_k_nearest_neighbour(training_data, training_labels, test_sample, K=1):
  distances = []
  for i, datapoint in enumerate(training_data):
    distance = get_euclidean_distance(
      datapoint.flatten(), test_sample.flatten()
    )  # converts to 1D and looks at euclidian distance between every training sample and the test sample
    distances.append((distance, i))

  knn_indeces = [j for _, j in sorted(distances)[:K]]
  knn_labels = [training_labels[j] for j in knn_indeces]  # sorting distances

  label_counter = {}
  for x in knn_labels:
    if x in label_counter:
      label_counter[x] += 1
    else:
      label_counter[x] = 1

  return max(
    label_counter, key=label_counter.get
  )  # returns label corresponding to max value


# KNN designed to be used with clusered
def get_k_nearest_neighbours_with_means(templates, test_sample, K=1):
  distances = []
  for label, cluster_centers in templates.items():
    for center in cluster_centers:
      distance = get_euclidean_distance(center, test_sample.flatten())
      distances.append((distance, label))

  knn_labels = [label for _, label in sorted(distances)[:K]]

  label_counter = {}
  for label in knn_labels:
    if label in label_counter:
      label_counter[label] += 1
    else:
      label_counter[label] = 1

  return max(label_counter, key=label_counter.get)


# om du vil teste KNN MED clustering, bruk denne
def evaluate_knn_with_means(templates, test_data, test_labels, K=1):
  timer = Timer("Evaluating KNN with clustering")
  timer.start()

  num_total_samples = test_data.shape[0]
  num_correct_predictions = 0
  predicted_labels = []

  for test_sample, true_label in zip(test_data, test_labels):
    label = get_k_nearest_neighbours_with_means(templates, test_sample, K)
    predicted_labels.append(label)
    if label == true_label:
      num_correct_predictions += 1

  accuracy = num_correct_predictions / num_total_samples

  cm = confusion_matrix(test_labels, predicted_labels)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt="d", cmap="rocket")  # endre farge hvis du vil
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.show()

  timer.stop()
  return accuracy, predicted_labels
