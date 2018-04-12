from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.matlab import mio5_params
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab
from data_convertion import MatConverter
import os


class DataExtractor:

    def __init__(self, path):
        self.path = path
        self.converter = MatConverter()
        self.data = self.converter.load_data(self.path)
        self.number_trials = self.data['data']['trial'].shape[0]
        self.number_time_indexes = len(self.time_series())
        self.number_channels = self.measures(trial=2, all_channels=True).shape[0]

    def time_series(self, trial=0, all_trials=False):
        '''Returns the time series associated to one or all the trials'''
        if all_trials:
            return self.data['data']['time']
        else:
            return self.data['data']['time'][trial]

    def measures(self, channel=0, trial=0, all_channels=False, all_trials=False, zero_padding=True):
        '''Return the measures associated to a specific trial and one or all the channels (captor)'''

        if all_trials and all_channels:
            return self.data['data']['trial']

        elif all_trials and not all_channels:
            # Returns nxm matrix with n = trials, m = time series
            matrix = np.zeros((self.number_trials, self.number_time_indexes))
            for i in range(self.number_trials):
                if zero_padding:
                    matrix[i, 0:len(self.data['data']['trial'][i][channel])] = self.data['data']['trial'][i][channel]
                else:
                    if len(self.data['data']['trial'][i][channel]) != self.number_time_indexes:
                        matrix = np.delete(matrix, matrix.shape[0] - 1, 0)
                    else:
                        matrix[i, :] = self.data['data']['trial'][i][channel]
            return matrix

        elif not all_trials and all_channels:
            return self.data['data']['trial'][trial]

        else:
            return self.data['data']['trial'][trial][channel]

    def labels(self, trial=0, all_trials=False):
        '''Returns the label for one or all the trials'''
        if all_trials:
            return self.data['data']['trialinfo'][:, 1]
        else:
            return self.data['data']['trialinfo'][trial, 1]

    def plot_measures(self, time_series, measures):
        '''Plots the measures'''
        time_min, time_max = min(time_series), max(time_series)
        meas_min, meas_max = min(measures), max(measures)
        plt.figure(1, figsize=(10, 5))
        plt.plot(time_series, measures)
        plt.xlim([time_min, time_max])
        plt.ylim([meas_min, meas_max])
        plt.show()

    def check_consistency(self, channel=95, trial=55):
        time = self.time_series(trial)
        measures = self.measures(channel, trial)
        trial = self.labels(trial)
        print('Data for the trial number ', trial, 'and the channel number ', channel)
        print('Type of trial (1, 2, 4, 5 or 6) : ', trial)
        print('Duration : ', time[-1]-time[0], 'seconds')
        print('Measures : \n \t Amplitude : ', max(measures) - min(measures))
        self.plot_measures(time, measures)

    def get_final_matrix(self):
        matrix = np.zeros((self.number_trials, self.number_time_indexes, self.number_channels))
        for i in range(self.number_trials):
            sequence_matrix = self.measures(trial=i, all_channels=True, all_trials=False, zero_padding=True)
            blank = np.zeros((1, sequence_matrix.shape[1], sequence_matrix.shape[0]))
            for row in range(sequence_matrix.shape[0]):
                blank[:, :, row] = sequence_matrix[row, :]
            print(blank)
            print('blank shape :', blank.shape)
            if sequence_matrix.shape != (1, self.number_time_indexes, self.number_channels):
                print('Bad input removed : time ', sequence_matrix.shape[1], ' and channels ', sequence_matrix.shape[2])
                matrix = np.delete(matrix, matrix.shape[0]-1, 0)
            else:
                matrix[i, :, :] = sequence_matrix
                print('seq ', i, ' appended')

        # Standardize on each channel
        matrix = matrix[:, :, matrix.shape[2]-4]
        for j in range(self.number_channels):
            channel = matrix[:, :, j]
            scaler = StandardScaler()
            scaler.fit(channel)
            standardized = scaler.transform(channel)
            matrix[:, :, j] = standardized
        print('final matrix : ', matrix)
        print('final matrix shape: ', matrix.shape)
        return matrix

    def get_labels_matrix(self, same_seq_lengths=False):
        labels = self.labels(all_trials=True)
        matrix = np.zeros((self.number_trials, 5))
        for i in range(self.number_trials):
            if same_seq_lengths:
                if self.get_one_matrix(i).shape != (1, self.number_time_indexes, self.number_channels):
                    matrix = np.delete(matrix, matrix.shape[0]-1, 0)
                else:
                    if labels[i] == 1.0:
                        matrix[i, :] = [1, 0, 0, 0, 0]
                    elif labels[i] == 2.0:
                        matrix[i, :] = [0, 1, 0, 0, 0]
                    elif labels[i] == 4.0:
                        matrix[i, :] = [0, 0, 1, 0, 0]
                    elif labels[i] == 5.0:
                        matrix[i, :] = [0, 0, 0, 1, 0]
                    elif labels[i] == 6.0:
                        matrix[i, :] = [0, 0, 0, 0, 1]
                    else:
                        print('Bad label input')
            else:
                if labels[i] == 1.0:
                    matrix[i, :] = [1, 0, 0, 0, 0]
                elif labels[i] == 2.0:
                    matrix[i, :] = [0, 1, 0, 0, 0]
                elif labels[i] == 4.0:
                    matrix[i, :] = [0, 0, 1, 0, 0]
                elif labels[i] == 5.0:
                    matrix[i, :] = [0, 0, 0, 1, 0]
                elif labels[i] == 6.0:
                    matrix[i, :] = [0, 0, 0, 0, 1]
                else:
                    print('Bad label input')
        print('labels :', matrix)
        print(matrix.shape)
        return matrix


class Augmentor:

    def subsample_data(self, labels_matrix,  sampling=600, stride=100, padding=0):
        number_samples_from_one_trial = 5
        #number_samples_from_one_trial = int((data_matrix.shape[1] - sampling + 2*padding)/stride)
        #new_matrix = np.zeros((number_samples_from_one_trial*data_matrix.shape[0], sampling, data_matrix.shape[2]))
        new_labels = np.zeros((number_samples_from_one_trial * labels_matrix.shape[0], labels_matrix.shape[1]))

        for i in range(labels_matrix.shape[0]):
            print('new trial')
            #trial = data_matrix[i, :, :]
            label = labels_matrix[i, :]
            #index = 0
            for j in range(number_samples_from_one_trial):
                #new_matrix[i*number_samples_from_one_trial+j, :, :] = trial[index:index+sampling, :]
                new_labels[i*number_samples_from_one_trial+j, :] = label
                #index += stride

        #print(new_matrix)
        print(new_labels)
        return new_labels


if __name__ == '__main__':

    '''data = np.zeros((1, 1221, 241))
    labels = np.zeros((1, 5))
    dirs_path = '././data/project_data/HCP/'
    for dir in os.listdir(dirs_path):
        data_path = '/MEG/Motort/tmegpreproc/'
        try:
            for file in os.listdir(dirs_path + str(dir) + data_path):
                if file.endswith('TFLA.mat'):
                    extractor = DataExtractor(dirs_path + str(dir) + data_path + '/' + file)
                    mat = extractor.get_final_matrix()
                    data = np.concatenate((data, mat), axis=0)
                    lab = extractor.get_labels_matrix(same_seq_lengths=False)
                    labels = np.concatenate(labels, lab)
        except:
            pass

    data = data[1:data.shape[0], :, :]
    labels = labels[1:, 5]
    np.savez('data_matrix.npz', data)
    np.savez('labels.npz', labels)'''

    extractor = DataExtractor('././data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')
    extractor.get_final_matrix()
    extractor.get_labels_matrix(same_seq_lengths=False)
