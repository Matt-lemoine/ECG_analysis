"""
Notes for me:
In the wfdb library, when you run __dict__['p_signal'], this return a numpy array of points along the different leads.
    So for example if you wanted to isolate one of the leads, you could do this by taking the following code:
    blah = record.__dict__['p_signal']
    I_lead = blah[0]

Note that the 'blah' above is a {t*100} x 12 matrix, because it store the leads as the columns and the mV as the rows.
The EKG records t*100 points for each lead. So that's 100 points for every second for every lead, there are 12 leads in our EKGs.
For the Brugada Dataset, there are 12 seconds of record for each patient, so the matrices of leads is (1200 x 12).
"""

# This is for visualizing the EKGs to make sure everything is up to snuff.

# import the WFDB package
import numpy as np
import pandas as pd
import matplotlib as plt
import wfdb
import csv


record = wfdb.rdrecord("Brugada_dataset/files/188981/188981")

wfdb.plot_wfdb(record =record, title='Patient{recrod}', figsize=(14,8), ecg_grids = 'all', sharex = True)
print(record.__dict__) ## Here you can add "__dict__['name of column you care about']" to highlight one specific area. Ex. p_signal, sig_name, shape
array = record.__dict__['p_signal']

csv_file_name = f'blah_blah.csv'
np.savetxt(f'{csv_file_name}', array, delimiter=",")

# def make_EKG_plot(path_to_file): #The input here, is the directions to the .hea and .dat files. (should be a string)
    
#     record = wfdb.rdrecord(path_to_file)

#     wfdb.plot_wfdb(record =record, title=f'Patient{path_to_file}', figsize=(14,8), ecg_grids = 'all', sharex = True)
    
# # You can get a list of all the patients in the 'files' file and then have a while loop cycle through that list. (this is for main_ekg.py)