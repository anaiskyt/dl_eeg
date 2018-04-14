import numpy as np
import mne
from dl_venv.dl_eeg.utils.extract_matrices import DataExtractor
from mne.preprocessing import ICA, create_ecg_epochs
from mne.channels import Montage
import hcp

trial = 2
extractor = DataExtractor('/home/anais/dl_project/dl_venv/dl_eeg/data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')
ch_names = list(extractor.channel_names())
ch_types = ['grad' for i in range(len(ch_names))]
sfreq = extractor.sampling_freq()  # Sampling frequency
times = extractor.time_series(trial=trial, all_trials=False)
positions = extractor.channel_position()

data = extractor.measures(trial=trial, all_trials=False, all_channels=True)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(data, info=info)

for i in range(241):
    print(i)
    raw.info['chs'][i]['loc'] = positions[i, :]


raw.plot_sensors(kind='topomap', ch_type='grad', block=True)

'''selection = np.zeros((1, 150))
selection[0, :] = range(150)
raw = mne.io.RawArray(extractor.channel_position(), info=None)
montage = Montage(extractor.channel_position(), extractor.channel_names(), 'biosemi256', selection=selection)
montage.plot()
'''