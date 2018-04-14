from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import BatchNormalization
import os



def create_model_lstm():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(1221, 242)))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def create_model_cnn():
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_dim=1221))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


def create_model_mlp():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=1221))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


print("loading data")
data = np.load('../data/data_matrix.npz')
labels = np.load('../data/labels.npz')
print("dataloaded")
data, labels = data['arr_0'], labels['arr_0']
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
tensorboard = TensorBoard(log_dir="logs", write_graph=True, write_images=True)
model = create_model_lstm()

model.fit(x_train, y_train,
          epochs=100,
          validation_split=0.2,
          shuffle=True,
          batch_size=64, callbacks=[tensorboard,
                                    EarlyStopping(patience=3, min_delta=0)])

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))