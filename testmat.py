



import os.path as op
from mne_hcp import hcp
import numpy as np
from mne_hcp.hcp import preprocessing as preproc

from mne.io import read_raw_bti
from scipy.io import loadmat




tmeg_file = "project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TEMG.mat"

tfla_file = "project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat"


trial_info_file = "project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_trialinfo.mat"



tmeg_data = loadmat(tmeg_file)

print(type(tmeg_data))

for cle in tmeg_data.keys():
    print(cle)
    print(type(tmeg_data[cle]))