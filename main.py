from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import save_model
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


"""s1 = loadmat('data/S1.mat')
s2 = loadmat('data/S2.mat')

x1_train, x1_test, y1_train, y1_test = get_data(s1)
x2_train, x2_test, y2_train, y2_test = get_data(s2)

x_train = np.concatenate((x1_train, x2_train), axis=0)
x_test = np.concatenate((x1_test, x2_test), axis=0)
y_train = np.concatenate((y1_train, y2_train), axis=0)
y_test = np.concatenate((y1_test, y2_test), axis=0)"""

data = np.load('106521_data_augmented.npz')
labels = np.load('106521_labels_augmented.npz')
data, labels = data['arr_0'], labels['arr_0']
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)


model = Sequential()
model.add(LSTM(256, input_shape=(600, 241)))
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score, acc = model.evaluate(x_test, y_test, batch_size=16)
print('Test score:', score)
print('Test accuracy:', acc)
model.save('simplest_lstm.h5')
