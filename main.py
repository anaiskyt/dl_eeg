from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import save_model
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


data = np.load('106521_data.npz')
labels = np.load('106521_labels.npz')
data, labels = data['arr_0'], labels['arr_0']
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)


model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(1221, 241)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score, acc = model.evaluate(x_test, y_test, batch_size=8)
print('Test score:', score)
print('Test accuracy:', acc)
model.save('simplest_lstm_50.h5')
