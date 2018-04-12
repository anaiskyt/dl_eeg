from keras.models import load_model
from dl_venv.dl_eeg.preprocessing import DataExtractor
import numpy as np


trial_nb = 175
data_trial = np.load('./data/matrices/106521_stand.npz')
trial = data_trial['arr_0'][trial_nb, :, :]
print(trial)
if trial.shape[0] != 1221:
    print('Pas bonne taille')
    data_trial.close()
else:
    trial = np.reshape(trial, (1, 1221, 241))
    print(f'Trial {trial_nb} loaded')
    data_trial.close()
    data_label = np.load('./data/matrices/106521_labels.npz')
    label = data_label['arr_0'][trial_nb, :]
    print(f'Label {trial_nb} is : ', label)
    data_label.close()

    model = load_model('./results/106521_basic.h5')
    print(model.predict(trial))
