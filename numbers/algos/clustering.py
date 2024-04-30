from sklearn.cluster import KMeans

# selveste clusteringen !! wow
def KMeansClustering(trainingData, trainingLabels, nClusters):
    kmeans = KMeans(n_clusters=nClusters, random_state=0) 
    clusterTemplates = {}

    for classLabel in np.unique(trainingLabels): 
        classData = trainingData[trainingLabels == classLabel].reshape(-1, (28*28)) 
        kmeans.fit(classData)
        clusterTemplates[classLabel] = kmeans.cluster_centers_
         
    return clusterTemplates