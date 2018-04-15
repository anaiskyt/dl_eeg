from keras.layers import Input, LSTM, RepeatVector, Dense, Dropout
from keras.models import Model, Sequential
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
import os
from time import time
from sklearn.model_selection import train_test_split


# load the data and split into train and test. Validation is retrieved automatically by Keras
data = np.load('../data/data_matrix.npz')
labels = np.load('../data/labels.npz')
data, labels = data['arr_0'], labels['arr_0']
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)


# we want turn 1221x242 dimensional data into 128-dimensional data
latent_dim = 128
timesteps = 1221
input_dim = 242

inputs = Input(shape=(timesteps, input_dim))

# we load the encoder and make it untrainable
encoder = Sequential()
encoder.add(LSTM(latent_dim))
encoder.load_weights('./weights_encoder.h5')
encoder.trainable = False
encoder = encoder(inputs)

# we create the rest of the model
layer_1 = Dense(64, input_shape=(128, 128), activation='relu')(encoder)
layer_1 = Dropout(0.5)(layer_1)
layer_2 = Dense(64, activation='relu')(layer_1)
layer_2 = Dropout(0.5)(layer_2)
layer_3 = Dense(5, activation='softmax')(layer_2)

complete_model = Model(inputs, layer_3)

complete_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# parameters for grid search
batch_size = [64, 128]
epochs = [50, 100]

for batch in batch_size:
    for epoch in epochs:
        directory = './logs/complete_b' + str(batch) + '_e_' + str(epoch)
        if not os.path.exists(directory):
            os.makedirs(directory)
        tensorboard = TensorBoard(log_dir=directory.format(time()), write_graph=True, write_images=True)

        complete_model.fit(x_train, y_train,
                           epochs=epoch,
                           batch_size=batch,
                           validation_split=0.5,
                           callbacks=[tensorboard, EarlyStopping(patience=3, min_delta=0)])
        score = complete_model.evaluate(x_test, y_test, batch_size=batch)

        #complete_model.save('complete_model.h5')
