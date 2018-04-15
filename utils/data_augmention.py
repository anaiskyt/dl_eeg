import numpy as np


class Augmentor:
    '''Augmenation of the data'''

    def subsample_data(self, data_matrix, labels_matrix,  sampling=800, stride=100, padding=0):
        number_samples_from_one_trial = int((data_matrix.shape[1] - sampling + 2*padding)/stride)
        print(number_samples_from_one_trial)
        new_data_matrix = np.zeros((data_matrix.shape[0]*number_samples_from_one_trial, sampling, data_matrix.shape[2]))
        new_labels = np.zeros((number_samples_from_one_trial * labels_matrix.shape[0], labels_matrix.shape[1]))

        for trial in range(data_matrix.shape[0]):
            for j in range(number_samples_from_one_trial):
                new_data_matrix[trial*number_samples_from_one_trial + j, :, :] = data_matrix[trial, j*stride:j*stride+sampling, :]

        for i in range(labels_matrix.shape[0]):
            print('Labels : new trial being subsampled')
            for j in range(number_samples_from_one_trial):
                new_labels[i*number_samples_from_one_trial+j, :] = labels_matrix[i, :]

        print(new_data_matrix.shape, new_labels.shape)
        return new_data_matrix, new_labels


if __name__ == '__main__':

    data = np.load('../data/data_matrix.npz')
    print('data loaded')
    labels = np.load('../data/labels.npz')
    data, labels = data['arr_0'], labels['arr_0']
    print('everythings loaded')

    augmentor = Augmentor()
    data_augmented, labels_augmented = augmentor.subsample_data(data, labels)
    np.savez('../data/data_matrix_augmented.npz', data_augmented)
    np.savez('../data/labels_augmented.npz', labels_augmented)