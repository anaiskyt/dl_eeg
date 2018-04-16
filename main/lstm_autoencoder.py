from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping

data = np.load('../data/data_matrix.npz')['arr_0']

latent_dim = 128
timesteps = 1221
input_dim = 242

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

sequence_autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

sequence_autoencoder.fit(data, data, epochs=100, batch_size=128, shuffle=True, validation_data=(data, data),
                         callbacks=[TensorBoard(log_dir='./logs/ae_128_batch_128_epoch_100', histogram_freq=0, write_graph=False), EarlyStopping(patience=3, min_delta=0)])

encoder.save_weights('weights_encoder_100.h5')
encoder.save('encoder_100.h5')
sequence_autoencoder.save('autoencoder_100.h5')
