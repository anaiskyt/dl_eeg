import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import save_model
from dl_eeg.preprocessing import get_data
import numpy as np
from scipy.io import loadmat
from pandas import DataFrame


def create_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(400, 10)))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

s1 = loadmat('data/S1.mat')
s2 = loadmat('data/S2.mat')

x1_train, x1_test, y1_train, y1_test = get_data(s1)
x2_train, x2_test, y2_train, y2_test = get_data(s2)

X = np.concatenate((x1_train, x2_train, x1_test, x2_test), axis=0)
Y = np.concatenate((y1_train, y2_train, y1_test, y2_test), axis=0)


# create model
model = KerasClassifier(build_fn=create_model, verbose=1)

# define the grid search parameters
batch_size = [16, 32]
epochs = [10, 50, 100]

param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

results = DataFrame(grid_result.cv_results_)
results.to_csv('simple_grid_search.csv')

print(results)
