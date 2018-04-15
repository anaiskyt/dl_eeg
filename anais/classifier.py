from sklearn.cluster import KMeans
import numpy as np

preds = np.load('predictions.npz')['arr_0']
labels = np.load('../data/labels.npz')['arr_0'][0:3000]


kmeans = KMeans(n_clusters=5, random_state=0).fit(preds)

dict_clusters = {}

for index, cluster_predicted in enumerate(kmeans.labels_):
    true_label = labels[index]
    if cluster_predicted not in dict_clusters.keys():
        dict_clusters[cluster_predicted] = [0, 0, 0, 0, 0]
        dict_clusters[cluster_predicted][true_label-1] = 1
    else:
        dict_clusters[cluster_predicted][true_label-1] +=1

print(dict_clusters)