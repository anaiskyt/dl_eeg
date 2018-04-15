import numpy as np
import mne
from dl_venv.dl_eeg.utils.extract_matrices import DataExtractor


# load main characteristic of the data : time length, mean number of channels, etc..
data_path = '/home/anais/dl_project/dl_venv/dl_eeg/data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat'
extractor = DataExtractor(data_path)

# define channels types and names, sampling frequency, sequence time length, channel positions
ch_names = list(extractor.channel_names())
ch_types = ['mag' for i in range(len(ch_names))]
sfreq = extractor.sampling_freq()
times = extractor.time_series(trial=3, all_trials=False)
positions = extractor.channel_position()

# load signals recreated from the autoencoder and adapt data
epochs_data = np.load('recreated_signals.npz')['arr_0'][0:50, :, :]
number_trials = epochs_data.shape[0]
reshaped = np.zeros((number_trials, 242, 1221))
for i in range(number_trials):
    transpose = np.transpose(epochs_data[i])
    reshaped[i, :, :] = transpose
epochs_data = reshaped[:, :241, :]

# create mne info
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
scalings = {'mag': 0.1 for i in range(len(ch_names))}

epochs = mne.EpochsArray(epochs_data, info=info)
picks = mne.pick_types(info, meg=True, eeg=False, eog=False)
epochs.plot(picks=picks, show=True, block=True, scalings=scalings)
