# Convert .mat data into dict

from scipy.io import loadmat
from scipy.io.matlab import mio5_params


class MatConverter:
    '''This class is used to convert the .mat data into python nested dictionaries.'''

    def load_data(self, path):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects

        from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
        '''
        data = loadmat(path, struct_as_record=False, squeeze_me=True)
        return self._check_keys(data)

    def _check_keys(self, d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], mio5_params.mat_struct):
                d[key] = self._todict(d[key])
        return d

    def _todict(self, matobj):
        '''
        A recursive function which constructs nested dictionaries from matobjects
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, mio5_params.mat_struct):
                d[strg] = self._todict(elem)
            else:
                d[strg] = elem
        return d


def analyse_nested_data(d, indent=0):
    '''This function returns the nested labels of the data contained in a dictionnary d.'''
    if isinstance(d, dict):
        for key in d.keys():
            print('\t'*indent, key)
            analyse_nested_data(d[key], indent+1)


if __name__ == '__main__':
    converter = MatConverter()

    #  load the .mat data and turn it into nested dictionnaries
    data = converter.load_data('../data/project_data/HCP/106521/MEG/Motort/tmegpreproc/106521_MEG_10-Motort_tmegpreproc_TFLA.mat')

    #  display the labels of the data available in the dataset
    analyse_nested_data(data)
    print(data['data']['cfg']['checkconfig'])
