#Creates a small data set for testing network

import pickle
import numpy as np
from utils.extract_matrices import DataExtractor
from utils.various import save_list, save2Dmat


extractor = DataExtractor('data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')

X, Y = [], []

for k in range(289):
    Xk = extractor.measures(all_channels=True, trial=k).T
    print(Xk.shape)

    X.append(Xk)
    Y.append(extractor.labels(trial=k))


X, Y = np.asarray(X), np.asarray(Y)


with open("small_data.pkl", "wb") as f:
    pickle.dump((X,Y), f)