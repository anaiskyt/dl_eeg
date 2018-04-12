import numpy as np

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


from utils import save_list

from utils import save_list, save2Dmat

from preprocessing import DataExtractor

import pickle


class Backend:

    def __init__(self):

        self.max_time_step = 1221
        self.number_channels = 241

        # create the model
        self.model = Sequential()
        self.model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True, input_shape=(self.max_time_step, self.number_channels)))
        self.model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))  # returns a sequence of vectors of dimension 32
        self.model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2,))  # return a single vector of dimension 32
        self.model.add(Dense(6, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(self.model.summary())

    def train(self, X_train, y_train, X_test, y_test):

        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)

    def evaluate(self, X_test, y_test):
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(7)
    max_review_length = 500
    with open("small_data.pkl", "rb") as f:
        (X,y) = pickle.load(f)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    backend = Backend()

    print("training")
    backend.train(X_train, y_train, X_test, y_test)
    print("evaluating")
    backend.evaluate(X_test, y_test)




