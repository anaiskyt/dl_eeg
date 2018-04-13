from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np


data = np.load('dl_eel/data/matrices/106521_data.npz')['arr_0']
data = np.reshape(data, (289, 1221, 241, 1))

input_img = Input(shape=(1221, 241, 1))  # adapt this if using `channels_first` image data format

encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
encoder = Model(input_img, encoded)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(data, data,
                epochs=50,
                batch_size=16,
                shuffle=True,
                validation_data=(data, data))

encoder.save_weights('encoder.h5')
