"""
This file is to normalize the EKG signal to:
60 bpm,
and extract about 8 peaks.
"""

import numpy as np
import matplotlib as plt
import wfdb
import csv

import Visualize_EKG as vekg

# You will want an input to be the normalization frequency.
# Big question: How do you stretch out or shrink down a signal to fit in xx bpm?

"""
The following function assumes only one lead at a time, so you will have to cycle through this function
    later to find specific valleys for different leads.

    The following function picks out the places of the valleys in a given lead.
"""

def find_valleys(ptf, lead):

    signal = vekg.get_EKG_leads(ptf, lead)

    test_valleys = []
    i = 0
    while i < len(signal): # This loop finds all the places where you have a low point
        if i+1 == len(signal):
            break
        elif signal[i] <= signal[i+1] and signal[i] <= signal[i-1] and signal[i]<=0:
            test_valleys.append(i)
            i += 1
        else:
            i +=1
    
    # Ok now valleys is every place where within one i there is a hill, so it could be something like ..., 0.1, 0.09, 0.1, ...

    average_test_valleys = np.mean(signal[test_valleys])
    std_test_valleys = np.std(signal[test_valleys])

    i = 0
    valleys = []
    while i < len(test_valleys):
        if signal[test_valleys[i]]<(average_test_valleys-std_test_valleys):
            valleys.append(test_valleys[i])
            i = i+1
        else:
            i = i+1

    return valleys
        

# def trim_EKG(ptf, lead_s, num_peaks):

# def normalize(ptf, lead_s, bpm):