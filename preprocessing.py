from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.matlab import mio5_params
from sklearn.model_selection import train_test_split
import matplotlib.pylab
from data_convertion import MatConverter


#  TO DO : refactor this function for HCP data
def get_data(s):
    data = np.zeros((160, 400, 10))
    labels = np.zeros((160, 4))
    for i in range(4):
        data[40*i:40*(i+1), :, :] = s['training_data'][0][i]
        labels[40*i:40*(i+1)] = [0 if j != i else 1 for j in range(4)]
    #  shuffle rows and divide into train and test
    return train_test_split(data, labels, test_size=0.2, shuffle=True)


def retrieve_and_count_elements(liste):
    values = {}
    for element in liste[:, 1]:
        if element not in values.keys():
            values[element] = 0
        else:
            values[element] += 1
    print(values)


class DataExtractor:

    def __init__(self, path):
        self.path = path
        self.converter = MatConverter()
        self.data = self.converter.load_data(self.path)

    def time_series(self, trial=0, all_trials=False):
        '''Returns the time series associated to one or all the trials'''
        if all_trials:
            return self.data['data']['time']
        else:
            return self.data['data']['time'][trial]

    def measures(self, channel=0, trial=0, all_channels=False):
        '''Return the measures associated to a specific trial and one or all the channels (captor)'''
        if all_channels:
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

    def get_one_matrix(self, trial=0):
        data = self.measures(trial=trial, all_channels=True)
        channels, time = data.shape
        matrix = np.zeros((1, time, channels))
        for i in range(channels):
            matrix[0, :, i] = data[i, :]
        return matrix

    def get_final_matrix(self):
        number_sequences = self.data['data']['trial'].shape[0]
        number_time_indexes = len(self.time_series())
        number_channels = self.measures(trial=2, all_channels=True).shape[0]
        matrix = np.zeros((number_sequences, number_time_indexes, number_channels))
        for i in range(number_sequences):
            sequence_matrix = self.get_one_matrix(i)[:, :, :number_channels]
            if sequence_matrix.shape != (1, number_time_indexes, number_channels):
                print('Bad input removed : time ', sequence_matrix.shape[1], ' and channels ', sequence_matrix.shape[2])
                matrix = np.delete(matrix, matrix.shape[0]-1, 0)
            else:
                matrix[i, :, :] = sequence_matrix
                print('seq ', i, ' appended')
        print(matrix)
        print(matrix.shape)
        return matrix

    def get_label(self):
        number_sequences = self.data['data']['trial'].shape[0]
        number_time_indexes = len(self.time_series())
        number_channels = self.measures(trial=2, all_channels=True).shape[0]
        labels = self.labels(all_trials=True)
        matrix = np.zeros((number_sequences, 5))
        for i in range(number_sequences):
            if self.get_one_matrix(i).shape != (1, number_time_indexes, number_channels):
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
        print(matrix)
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
<<<<<<< HEAD
    #extractor = DataExtractor('./data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')
    augmentor = Augmentor()
=======
    extractor = DataExtractor('data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')
>>>>>>> 7fcd26568022634b945ad4667551e96efe10b6aa

    #  Overview of the data for a given channel and trial
    #extractor.check_consistency(180, 360)

    """labels = np.load('106521_labels.npz')
    labels = labels['arr_0']
    #labels = labels['arr_0']
    #print(labels.shape)
    new_lab = augmentor.subsample_data(labels)
    np.savez('106521_labels_augmented.npz', new_lab)"""

    labels = np.load('106521_labels_augmented.npz')
    labels = labels['arr_0']
    print(labels.shape)