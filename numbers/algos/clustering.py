import numpy as np

from sklearn.cluster import KMeans

# selveste clusteringen !! wow
def get_means_clustering_templates(training_data, training_labels, num_clusters):
    means = KMeans(n_clusters=num_clusters, random_state=0) 
    templates = {}

    for class_label in np.unique(training_labels): 
        class_data = training_data[training_labels == class_label].reshape(-1, (28*28)) 
        means.fit(class_data)
        templates[class_label] = means.cluster_centers_

    return templates