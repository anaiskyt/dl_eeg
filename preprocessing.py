from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.matlab import mio5_params
from sklearn.model_selection import train_test_split


def print_mat_nested(d, indent=0, nkeys=0):
    """Pretty print nested structures from .mat files
    Inspired by: `StackOverflow <http://stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python>`_
    """

    # Subset dictionary to limit keys to print.  Only works on first level
    if nkeys > 0:
        d = {k: d[k] for k in d.keys()}  # Dictionary comprehension: limit to first nkeys keys.

    if isinstance(d, dict):
        for key, value in d.items():  # iteritems loops through key, value pairs
            print('\t' * indent + 'Key: ' + str(key))
            print_mat_nested(value, indent + 1)

    if isinstance(d, np.ndarray) and d.dtype.names is not None:  # Note: and short-circuits by default
        for n in d.dtype.names:  # This means it's a struct, it's bit of a kludge test.
            print('\t' * indent + 'Field: ' + str(n) + 'Value: ' + str(d[n]))
            print_mat_nested(d[n], indent + 1)


def dtype_shape_str(x):
    """ Return string containing the dtype and shape of x."""
    return str(x.dtype) + " " + str(x.shape)


def new_loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict



def get_data(s):
    data = np.zeros((160, 400, 10))
    labels = np.zeros((160, 4))
    for i in range(4):
        data[40*i:40*(i+1), :, :] = s['training_data'][0][i]
        labels[40*i:40*(i+1)] = [0 if j != i else 1 for j in range(4)]
    #  shuffle rows and divide into train and test
    return train_test_split(data, labels, test_size=0.2, shuffle=True)




"""electrode = range(data[0][0].shape[0])
time = range(data[0][0].shape[1])
print(time)
for i in electrode:
    plt.plot(time, data[0][0][i, :])

plt.show()"""

