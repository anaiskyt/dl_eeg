from keras.models import load_model
import numpy as np

data = np.load('../data/data_matrix.npz')['arr_0']
labels = np.load('../data/labels.npz')['arr_0']
print('True label : ', labels[55])

model = load_model('106521_basic.h5')
predictions = model.predict(data['data']['trial'][55])

print('First prediction:', predictions[0])
print('Accuracy : ', predictions[1])