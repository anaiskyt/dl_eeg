import numpy as np
import mne
from dl_venv.dl_eeg.utils.extract_matrices import DataExtractor

trial = 2
extractor = DataExtractor('/home/anais/dl_project/dl_venv/dl_eeg/data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')
ch_names = list(extractor.channel_names())
ch_types = ['mag' for i in range(len(ch_names))]
sfreq = extractor.sampling_freq()  # Sampling frequency
times = extractor.time_series(trial=trial, all_trials=False)
positions = extractor.channel_position()
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
scalings = {'mag': 1 for i in range(len(ch_names))}

epochs_data = np.load('recreated_signals.npz')['arr_0'][0, :, :]

print(epochs_data)
number_trials = epochs_data.shape[0]
reshaped = np.zeros((number_trials, 242, 1221))
for i in range(number_trials):
    transpose = np.transpose(epochs_data)
    #print(transpose.shape)
    '''if epochs_data[i].shape[1] < extractor.number_time_indexes:
        print('discarded')
        pass
    else:'''
    reshaped[i, :, :] = transpose
epochs_data = reshaped[:, :241, :]
print(epochs_data.shape)

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

epochs = mne.EpochsArray(epochs_data, info=info)

picks = mne.pick_types(info, meg=True, eeg=False, eog=False)
epochs.plot(picks=picks, show=True, block=True, scalings=scalings)