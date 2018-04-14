
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.matlab import mio5_params
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab
from utils.data_convertion import MatConverter
import os
from utils.extract_matrices import DataExtractor


data = np.zeros((1, 1221, 242))
labels = np.zeros((1, 5))
dirs_path = '../data/project_data/HCP/'
for dir in os.listdir(dirs_path):
    if not dir.startswith('.'):
        data_path = '/MEG/Motort/tmegpreproc/'
        for file in os.listdir(dirs_path + str(dir) + data_path):
            if file.endswith('TFLA.mat'):
                extractor = DataExtractor(dirs_path + '/' + str(dir) + data_path + '/' + file)
                mat = extractor.get_final_matrix()
                data = np.concatenate((data, mat), axis=0)
                lab = extractor.get_labels_matrix(same_seq_lengths=False)
                lab = lab.astype(int)
                labels = np.concatenate((labels, lab), axis=0)

data = data[1:data.shape[0], :, :]
labels = labels[1:, :]
np.savez('../data/data_matrix.npz', data)
np.savez('../data/labels.npz', labels)