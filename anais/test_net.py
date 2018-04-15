from keras.models import load_model
import numpy as np

data = np.load('../data/data_matrix.npz')['arr_0'][0:3000]
#data = np.reshape(data, (3, 1221, 242))
#labels = np.load('../data/labels.npz')['arr_0']
#print('True label : ', labels)

model = load_model('encoder.h5')
predictions = model.predict_on_batch(data)
np.savez('preditions_ae.npz', predictions) 
#print('First prediction:', predictions)
