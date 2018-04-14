from keras.models import load_model
import numpy as np

data = np.load('../data/data_matrix.npz')['arr_0'][55:58]
#data = np.reshape(data, (3, 1221, 242))
#labels = np.load('../data/labels.npz')['arr_0']
#print('True label : ', labels[55:58])

model = load_model('encoder.h5')
predictions = model.predict_on_batch(data)

print('First prediction:', predictions)
