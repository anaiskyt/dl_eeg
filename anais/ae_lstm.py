from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
import numpy as np
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, RMSprop

data = np.load('../data/data_matrix.npz')['arr_0']

latent_dim = 50
timesteps = 1221
input_dim = 242

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

sgd = SGD(lr=5, momentum=1.5, decay=1e-6, nesterov=False)
rmsprop = RMSprop(lr=1, rho=0.9)

sequence_autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
sequence_autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True, validation_data=(data, data),
                         callbacks=[TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=False)])
