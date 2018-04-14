from keras.layers import Input, LSTM, RepeatVector, Dense, Dropout
from keras.models import Model, Sequential
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
import os
from time import time
from sklearn.model_selection import train_test_split


data = np.load('../data/data_matrix.npz')
labels = np.load('../data/labels.npz')
data, labels = data['arr_0'], labels['arr_0']
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

latent_dim = 128
timesteps = 1221
input_dim = 242

inputs = Input(shape=(timesteps, input_dim))

encoder = Sequential()
encoder.load_weights('./weights_encoder.h5', by_name=True)
encoder.trainable = False

encoded = Model(inputs, encoder)

layer_1 = Dense(64, input_shape=(128, 128), activation='relu')(encoded)
layer_1 = Dropout(0.5)(layer_1)
layer_2 = Dense(64, activation='relu')(layer_1)
layer_2 = Dropout(0.5)(layer_2)
layer_3 = Dense(5, activation='softmax')(layer_2)

complete_model = Model(encoded, layer_3)


complete_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


batch_size = [128, 256]
epochs = [10, 50, 100]

for batch in batch_size:
    for epoch in epochs:
        directory = './logs/complete_b' + str(batch) + '_e_' + str(epoch)
        if not os.path.exists(directory):
            os.makedirs(directory)
        tensorboard = TensorBoard(log_dir=directory.format(time()), write_graph=True, write_images=True)

        complete_model.fit(x_train, y_train,
                           epochs=10,
                           batch_size=128,
                           callbacks=[tensorboard, EarlyStopping(patience=3, min_delta=0)])
        score = complete_model.evaluate(x_test, y_test, batch_size=128)

        complete_model.save('complete_model.h5')