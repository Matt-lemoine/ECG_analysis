import numpy as np
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
            return print(error_message)
        i = i+1

    # Now we pick out the desired leads using the positions list.

    p_signals = record.__dict__['p_signal'][:, positions]

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

    # Now I want to plot all the leads on the same plot.

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

    all_info =  wfdb.rdrecord(ptf).__dict__

    return print(all_info)
