# Test network on exemples

from keras.models import load_model
import numpy as np
from sklearn.cluster import KMeans


# load the data
data = np.load('../data/data_matrix.npz')['arr_0'][0:3000]
labels = np.load('../data/labels.npz')['arr_0'][0:3000]

# load the encoder
model = load_model('encoder.h5')

# return the 128-dimensional representations predicted of the data
predictions = model.predict_on_batch(data)

print('First prediction:', predictions)

# KMeans algorithm on the representation of the data
kmeans = KMeans(n_clusters=5, random_state=0).fit(predictions)

dict_clusters = {}

for index, cluster_predicted in enumerate(kmeans.labels_):
    true_label = labels[index]
    if cluster_predicted not in dict_clusters.keys():
        dict_clusters[cluster_predicted] = [0, 0, 0, 0, 0]
        dict_clusters[cluster_predicted][true_label-1] = 1
    else:
        dict_clusters[cluster_predicted][true_label-1] += 1

print(dict_clusters)
