"""
Notes on this file:
    There are three functions in this file:

    get_EKG_leads(ptf, lead_s)
        Inputs:
            path to file (ptf) which should be given as directions from the current folder you are in.
            leads (lead_s) you want to isolate, which should be given as an array of strings out of the following list.
                lead_s = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                Note: In WFDB, these leads are called the 'sig_name'. So, if you wanted to call these from WFDB, you'd use the command
                    record.__dict__['sig_name']
        Output:
            An numpy array with the lead(s) that you want isolated
    This function picks out the specific leads you are wanting as a numpy array.
    
    plot_ekg(ptf, lead_s)
        Inputs:
            path to file (ptf) which should be given as directions from the current folder you are in.
            leads (lead_s) you want to plot, which should be given as an array of strings out of the following list.
                lead_s = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        Output:
            A matplotlib image of the leads you want plotted
    This function plots the specific leads you are interested in.

    get_all_info(ptf)
        Input:
            path to file (ptf)
        Output:
            A list of infromation about your EKG. This is exactly the 'record.__dict__' function from wfdb.
    This function gets all the information about your EKG and prints it.


Notes on how WFDB is used here:

In the wfdb library, when you run __dict__['p_signal'], this return a numpy array of points along the different leads.
    So for example if you wanted to isolate one of the leads, you could do this by taking the following code:
    blah = record.__dict__['p_signal']
    I_lead = blah[0]

    Note that the 'blah' above is a {t*100} x 12 matrix, because it store the leads as the columns and the mV as the rows.
    The EKG records t*100 points for each lead. So that's 100 points for every second for every lead, there are 12 leads in our EKGs.
    For the Brugada Dataset, there are 12 seconds of record for each patient, so the matrices of leads is (1200 x 12).
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import wfdb
import csv

def get_EKG_leads(ptf, lead_s):

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

def plot_ekg(ptf, lead_s, naming_things):

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

    # # # Ok now I want to plot all the leads on the same plot.

    # I need this little piece to name the y-axis of the subfigures.
    i = 0
    sig_names = []
    sig_units_labels = []
    while i < len(positions):
        blah = record.sig_name[positions[i]]
        sig_names.append(blah)
        sig_units_labels.append('mV')
        i = i+1

    return wfdb.plot_items(signal=record.p_signal[:, positions],
                    ann_samp=None, 
                    ann_sym=None, 
                    fs=None,
                    sig_name=sig_names, 
                    sig_units=sig_units_labels,
                    xlabel=None, 
                    ylabel=None, 
                    title=f'Patient {naming_things}', 
                    sig_style=[''],
                    ann_style=['r*'], 
                    ecg_grids=[], 
                    figsize=(14,8),
                    sharex=True, 
                    sharey=False, 
                    return_fig=False, 
                    return_fig_axes=False)

def get_all_info(ptf):

    record =  wfdb.rdrecord(ptf)

    all_info = record.__dict__

    return print(all_info)
