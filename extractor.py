
import os.path as op
from mne_hcp import hcp
import numpy as np
from mne_hcp.hcp import preprocessing as preproc

from mne.io import read_raw_bti
from scipy.io import loadmat



class Extractor:

    def __init__(self, folder_path):

        self.folder_path = folder_path
        self.data_type = 'task_motor'

    def print_subject_raw(self, subject):


        data = hcp.read_raw(subject=subject, data_type=self.data_type, hcp_path=folder_path)

        data = data.to_data_frame()

        print(type(data))

        print(data[1][1][1])

        #data = read_raw_bti(folder_path + "/104012/unprocessed/MEG/10-Motort/4D/c,rfDC")
        #print(type(data))

        #data = np.fromfile(folder_path + "/104012/unprocessed/MEG/10-Motort/4D/c,rfDC")
        print(type(data))


        print(data.size)


    def print_epochs(self, subject):

        data = hcp.read_epochs(subject=subject, data_type=self.data_type, hcp_path=folder_path)

        print(type(data))



if __name__ == "__main__":
    """folder_path = "project_data/HCP"

    extractor = Extractor(folder_path)

    #subject = '104012'

    #extractor.print_subject_raw(subject)

    subject = "106521"

    extractor.print_epochs(subject)"""

    data = loadmat("/Users/aubay/Documents/Cours_Centrale_3A/Deep_Learning/projetfinal/dl_eeg/project_data/HCP/106521/MEG/Motort/eravg/106521_MEG_Motort_eravg_[LM-TEMG-LF]_[BT-diff]_[MODE-mag].mat")



    print(data.size)

