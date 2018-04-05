#Creates a small data set for testing network

import pickle
import numpy as np
from preprocessing import DataExtractor

from utils import save_list

extractor = DataExtractor('data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')

#  Overview of the data for a given channel and trial
#extractor.check_consistency(180, 360)


"""ts = extractor.time_series()
print(ts)
print(type(ts))

ts1 = np.concatenate((ts, np.array([0])))
ts2 = np.concatenate((np.array([0]), ts))

tsv = ts2 - ts1


save_list(ts, "ts")

save_list(tsv, "tsv")"""


data = np.zeros((3,4))





with open("small_data.pkl", "wb") as f:
    pickle.dump(data, f)