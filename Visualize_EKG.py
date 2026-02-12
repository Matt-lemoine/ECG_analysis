# This is for visualizing the EKGs to make sure everything is up to snuff.

# import the WFDB package
import numpy as np
import pandas as pd
import matplotlib as plt
import wfdb


record = wfdb.rdrecord("Brugada_dataset/files/188981/188981")

wfdb.plot_wfdb(record =record, title='Patient{recrod}', figsize=(14,8), ecg_grids = 'all', sharex = True)

# def make_EKG_plot(path_to_file): #The input here, is the directions to the .hea and .dat files. (should be a string)
    
#     record = wfdb.rdrecord(path_to_file)

#     wfdb.plot_wfdb(record =record, title='Patient{recrod}', figsize=(14,8), ecg_grids = 'all', sharex = True)
    
# # You can get a list of all the patients in the 'files' file and then have a while loop cycle through that list. (this is for main_ekg.py)