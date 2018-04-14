import numpy as np
import mne
from dl_venv.dl_eeg.utils.extract_matrices import DataExtractor
from mne.preprocessing import ICA, create_ecg_epochs
from mne.channels import Montage
import hcp

trial = 2
extractor = DataExtractor('/home/anais/dl_project/dl_venv/dl_eeg/data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')
ch_names = list(extractor.channel_names())
ch_types = ['mag' for i in range(len(ch_names))]
sfreq = extractor.sampling_freq()  # Sampling frequency
times = extractor.time_series(trial=trial, all_trials=False)
positions = extractor.channel_position()
scalings = {'mag': 1e-12 for i in range(len(ch_names))}

epochs_data = extractor.measures(all_trials=True, all_channels=True)
reshaped = np.zeros((384, 241, 1221))
for i in range(384):
    if epochs_data[i].shape[1] < extractor.number_time_indexes:
        pass
    else:
        reshaped[i, :, :] = epochs_data[i]
epochs_data = reshaped
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

epochs = mne.EpochsArray(epochs_data, info=info)

#raw.plot_sensors(kind='topomap', ch_type='mag', block=True)

picks = mne.pick_types(info, meg=True, eeg=False, eog=False)
#epochs.plot(picks=picks, show=True, block=True, scalings=scalings)

data = epochs_data[:, :, 50]
data = np.transpose(data)
print(data.shape)
raw = mne.io.RawArray(data, info=info)
raw.filter(.5, 25, fir_design='firwin') 
evokeds = mne.EvokedArray(raw, info=info, tmin=-0.2, comment='Arbitrary')

for i in range(241):
    print(i)
    evokeds.info['chs'][i]['loc'] = positions[i, :]

#evokeds.plot(spatial_colors=True, gfp=True, picks=picks)

evokeds.plot_topomap(times=times)


'''selection = np.zeros((1, 150))
selection[0, :] = range(150)
raw = mne.io.RawArray(extractor.channel_position(), info=None)
montage = Montage(extractor.channel_position(), extractor.channel_names(), 'biosemi256', selection=selection)
montage.plot()
'''