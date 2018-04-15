import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from dl_venv.dl_eeg.utils.data_convertion import MatConverter
import os


class DataExtractor:
    '''This class is used to extract all information from a data file :
    * measures
    * time series
    * sensors (=channels) positions
    etc...'''

    def __init__(self, path):
        self.path = path
        self.converter = MatConverter()
        self.data = self.converter.load_data(self.path)
        self.number_trials = self.data['data']['trial'].shape[0]
        self.number_time_indexes = len(self.time_series())
        #self.number_channels = len(self.data['data']['label'])
        self.number_channels = 246

    def sampling_freq(self):
        return self.data['data']['fsample']

    def channel_names(self):
        list_chan = [channel for channel in self.data['data']['label']]
        return list_chan

    def channel_position(self):
        ''' Returns 271*3 matrix containing all the channels 3D coordinates'''
        return self.data['data']['grad']['chanpos']

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
                if self.data['data']['trial'][i].shape[0] <= channel:
                    print('Some channels were removed from the trial data')
                    pass
                else:
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
        matrix = np.zeros((self.number_trials, self.number_time_indexes, self.number_channels-4))
        for i in range(self.number_channels-4):
            channel_matrix = self.measures(channel=i, all_channels=False, all_trials=True, zero_padding=True)
            scaler = StandardScaler()
            scaler.fit(channel_matrix)
            standardized = scaler.transform(channel_matrix)
            print(standardized)
            '''blank = np.zeros((channel_matrix.shape[0], channel_matrix.shape[1], 1))
            for row in range(channel_matrix.shape[0]):
                blank[row, :, 0] = standardized[row, :]
            print(blank)
            print('blank shape :', blank.shape)'''
            if standardized.shape != (self.number_trials, self.number_time_indexes):
                print('Bad input removed : time ', standardized.shape[1], ' and channels ', standardized.shape[0])
                matrix = np.delete(matrix, matrix.shape[0]-1, 0)
            else:
                matrix[:, :, i] = standardized
                print('seq ', i, ' appended')

        print('final matrix : ', matrix)
        print('final matrix shape: ', matrix.shape)
        return matrix

    def get_labels_matrix(self, same_seq_lengths=False):
        labels = self.labels(all_trials=True)
        matrix = np.zeros((self.number_trials, 5))
        for i in range(self.number_trials):
            if same_seq_lengths:
                if self.measures(trial=i, all_channels=True, all_trials=False).shape != (1, self.number_time_indexes, self.number_channels):
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


if __name__ == '__main__':

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


