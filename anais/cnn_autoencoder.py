from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split


# load the data and split into train and test. Validation is retrieved automatically by Keras
data = np.load('../data/data_matrix.npz')['arr_0']
data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
x_train, x_test, y_train, y_test = train_test_split(data, data, test_size=0.2, shuffle=True)

timesteps = 1221
input_dim = 242

input_img = Input(shape=(timesteps, input_dim, 1))
x = Conv2D(256, (3, 3), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(256, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(x_train, y_train, epochs=10, batch_size=128, shuffle=True, validation_data=(x_test, y_test),
                         callbacks=[TensorBoard(log_dir='./logs/ae_128_batch_128_epoch_10_lr01', histogram_freq=0, write_graph=False), EarlyStopping(patience=3, min_delta=0)])

autoencoder.save_weights('weights_cnn_encoder.h5')
autoencoder.save('cnn_encoder.h5')
