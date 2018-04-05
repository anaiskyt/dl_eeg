import os.path as op
from mne_hcp import hcp
import mne
from mne_hcp.hcp import preprocessing as preproc

##############################################################################
# we assume our data is inside a designated folder under $HOME
storage_dir = "project_data"
hcp_path = op.join(storage_dir, 'HCP')
recordings_path = op.join(storage_dir, 'hcp-meg')
subjects_dir = op.join(storage_dir, 'hcp-subjects')
subject = '104012'  # our test subject
data_type = 'task_motor'
run_index = 0

##############################################################################
# Let's get the evoked data.

data = mne.io.read_raw_bti()

info = hcp.read_raw(subject= subject, data_type= data_type, hcp_path= hcp_path)

print(info[1][1][1])
print(type(info))


"""hcp_evokeds = hcp.read_evokeds(onset='stim', subject=subject, data_type=data_type, hcp_path='hcp_path')
print(hcp_evokeds)
for evoked in hcp_evokeds:
    if not evoked.comment == 'Wrkmem_LM-TIM-face_BT-diff_MODE-mag':
        continue
##############################################################################
# In order to plot topographic patterns we need to transform the sensor
# positions to MNE coordinates one a copy of the data.
# We're not using this these transformed data for any source analyses.
# These take place in native BTI/4D coordinates.

evoked_viz = evoked.copy()
preproc.map_ch_coords_to_mne(evoked_viz)
evoked_viz.plot_joint()"""
