from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import BatchNormalization
import os


# define the grid search parameters
batch_size = [16]
epochs = [10]

data = np.load('106521_stand.npz')
labels = np.load('106521_labels.npz')
data, labels = data['arr_0'], labels['arr_0']
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)


def create_model_lstm():
    model = Sequential()
    model.add(LSTM(256, input_shape=(1221, 241)))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='mse',
                  optimizer='adam',
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


for batch in batch_size:
    for epoch in epochs:
        directory = './logs/lstm_1_layer_' + str(batch) + '_epoch' + str(epoch)
        if not os.path.exists(directory):
            os.makedirs(directory)
        tensorboard = TensorBoard(log_dir=directory.format(time()), write_graph=True, write_images=True)
        model = create_model_lstm()
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  epochs=epoch,
                  batch_size=batch, callbacks=[tensorboard])
        score = model.evaluate(x_test, y_test, batch_size=128)
        model.save('106521_basic.h5')