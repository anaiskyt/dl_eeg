from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.matlab import mio5_params
from sklearn.model_selection import train_test_split
import matplotlib.pylab
from dl_eeg.data_convertion import MatConverter


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
    for element in test[:, 1]:
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

    def time_series(self, trial=0):
        '''Returns the time series associated to a specific trial'''
        return self.data['data']['time'][trial]

    def measures(self, channel=0, trial=0):
        '''Return the measures associated to a specific trial and a specific channel (captor)'''
        return self.data['data']['trial'][trial][channel]

    def labels(self, trial=0):
        '''Returns the label for a given trial'''
        return self.data['data']['trialinfo'][trial, 1]

    def plot_measures(self, time_series, measures):
        '''Plots the measures'''
        time_min, time_max = min(time_series), max(time_series)
        meas_min, meas_max = min(measures), max(measures)
        plt.figure(1, figsize=(50, 30))
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


if __name__ == '__main__':
    extractor = DataExtractor('./data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')

    #  Overview of the data for a given channel and trial
    extractor.check_consistency(180, 360)

    # Create matrix with all the data and labels
