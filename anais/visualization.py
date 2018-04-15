import numpy as np
import mne
from dl_venv.dl_eeg.utils.extract_matrices import DataExtractor
from mne.preprocessing import ICA, create_ecg_epochs
from mne.channels import Montage
import hcp

trial = 2
extractor = DataExtractor('/home/anais/dl_project/dl_venv/dl_eeg/data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')
sfreq = extractor.sampling_freq()  # Sampling frequency
times = extractor.time_series(trial=trial, all_trials=False)
ch_names = extractor.channel_names()
ch_names = list(ch_names)
ch_types = ['grad' for i in range(len(ch_names))]
print(len(ch_names))
scalings = {key:1e-15 for key in ch_names}

data = np.load('recreated_signals.npz')['arr_0']
'''reshaped = np.zeros((4, 241, 1221))
reshaped = []
for i in range(4):
    if data[i].shape[1] < extractor.number_time_indexes:
        pass
    else:
        #reshaped[i, :, :] = data[i]
        reshaped.append(data[i])'''

info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
raw = mne.io.RawArray(data, info=info)
picks = mne.pick_types(info, meg=True, eeg=False, misc=False)

raw.filter(1, 30, fir_design='biosemi256')

events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.2, tmax=0.5)

ica = ICA(n_components=0.95, method='fastica').fit(epochs)

ecg_epochs = create_ecg_epochs(epochs, tmin=-.5, tmax=.5)
ecg_inds, scores = ica.find_bads_ecg(ecg_epochs)

ica.plot_components(ecg_inds)