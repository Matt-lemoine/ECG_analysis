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

# path_to_file = "Brugada_dataset/files/188981/188981"

# record = wfdb.rdrecord(path_to_file)




# wfdb.plot_wfdb(record =record, title='Patient{recrod}', figsize=(14,8), ecg_grids = 'all', sharex = True) # This plots all 12 leads.
# print(record.__dict__) ## Here you can add "__dict__['name of column you care about']" to highlight one specific area. Ex. p_signal, sig_name, shape
# array = record.__dict__['p_signal'] # This isolates the p_signals from the record, this is the 1200x12 matrix with the 12 leads as the columns.

# wfdb.plot_wfdb(signal = record.p_signal[:,0], title='Patient{recrod}')

# csv_file_name = f'blah_blah.csv'
# np.savetxt(f'{csv_file_name}', array, delimiter=",")

# Plot only the first column (index 0)
# p_signal is a numpy array [samples, channels]



# wfdb.plot_items(signal=record.p_signal[:,0], title=f'Patient{path_to_file}')















"""
As I understand it, we only care about V1 - V3 leads. So I want to isolate these 3 leads.

Note to self: Later we can add back a few other leads to look for patterns in those leads as well, but for now let's focus on the ones that are used
    for diagnosis.
"""

# # # Ok, I'd like to make a function that picks out the p_signal for a specific lead(s).
# Your input: Path_to_file (PTF), lead(s)
    # PTF should be given from the folder you are coding in.
    # lead(s) should be given as an array of strings like I have in the def get_EKG_leads
# Your output: a numpy array with the specific lead(s) asked for.

def get_EKG_leads(ptf, lead_s):

    # I want to input the lead(s) as an array of the title of the leads so if you wanted all twelve it would look like:
    # lead_s = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
    # Note that in __dict__ these are called 'sig_name'

    record =  wfdb.rdrecord(ptf)
    signal_names = record.__dict__['sig_name']

    # Now I want to cycle through the lead_s and pick out the position in the p_signal which will match up with the place in the sig_name list.

    i = 0
    positions = []
    while i<len(lead_s):
        try:
            lead_s[i] in signal_names
            position = signal_names.index(lead_s[i])
            positions.append(position)
        except:
            error_message = "You do not have the leads labeled correctly, refer to input information for how to input the leads."
            return error_message
        i = i+1

    # Ok, so now you have the leads you want and the positions in the positions array.

    p_signals = record.__dict__['p_signal']

    p_signals = p_signals[:, positions]

    return p_signals























# def make_EKG_plot(path_to_file): #The input here, is the directions to the .hea and .dat files. (should be a string)
    
#     record = wfdb.rdrecord(path_to_file)

#     wfdb.plot_wfdb(record =record, title=f'Patient{path_to_file}', figsize=(14,8), ecg_grids = 'all', sharex = True)
    
# # You can get a list of all the patients in the 'files' file and then have a while loop cycle through that list. (this is for main_ekg.py)