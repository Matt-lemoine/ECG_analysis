"""
This file is to normalize the EKG signal. The functions here help normalize you EKGs so that they tau and M that are used 
    can be standard across all the EKGs we want to analyze.
"""

import numpy as np
import matplotlib as plt
import wfdb
import csv

import Visualize_EKG as vekg

"""
In the following function (find_valleys), we are looking for places where the signal dips down toward -1 mV. Because we are looking for places 
    where our signal dips down, we are only looking at one lead at a time. Furthermore, this function only takes in leads V1, V2, and V3. This
    will help us be able to trim down our EKG signals so that we can normalize our signals.
"""

def find_valleys(ptf, lead):

    lenlead = len(lead)

    if lenlead == 1:
        signal = vekg.get_EKG_leads(ptf, lead)
    else:
        error = 'You can only import one lead at a time for find_valleys.'
        return print(error)

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
            i += 1
        else:
            i += 1

    # Now you need to double check that you didn't accedientally pick up an extra valley inside another valley.
    # The 15 here is arbitrary, it corresponds to 15 centisecond in the ekg reading. It does not make sense that 
        # there would be 2 valleys withing 15 centiseconds of each other.

    i = 0
    while i < len(valleys)-1:
        if valleys[i+1] - valleys[i] > 15:
            i += 1
        elif valleys[i+1] - valleys[i] <= 15:
            signal_i = signal[valleys[i]]
            signal_i1 = signal[valleys[i+1]]
            if signal_i <= signal_i1:
                del valleys[i+1]
                i += 1
            elif signal_i > signal_i1:
                del valleys[i]
                i += 1
    
    return valleys

"""
In the following function (trim_EKG), we take a given signal and a single lead that we are interested in and trim down the 
    EKG signal to be the middle-most 6 peaks in our reading. This function only works for leads V1, V2, and V3. The num_peaks + 2
    corresponds to num_valleys, because we want a valley on both sides of our middle-most num_peaks. (For example, if we are looking at
    the middle-most 6 peaks, then we are looking at the 8 middle-most valleys.) 

    We make the assumption that if our signal is not even, then we will trim more off the end of the signal than the beginning. The if-else piece
    at the end of this function finds how much to trim off the front/back and returns the valleys that are in our trimmed signal. Then after this 
    if-else piece we perform the trimming and get our trimmed signal.
"""

def trim_EKG(ptf, lead):

    num_peaks = 6  # To edit the normalized num_peaks, edit here.

    lenlead = len(lead)

    if lenlead == 1:
        signal = vekg.get_EKG_leads(ptf, lead)
    else:
        error = 'You can only import one lead at a time for trim_EKG.'
        return print(error)

    valleys = find_valleys(ptf, lead)
    num_valleys = len(valleys)

    if num_valleys < num_peaks + 2:
        error = f"You need at least two more peaks in your reading than your valleys in order to trim, you have {num_valleys} valleys"
        return print(error)
    else:
        diff = num_valleys - num_peaks
        front_trim = diff // 2
        vall = valleys[front_trim - 1 : front_trim + num_peaks + 1]

    new_range = np.arange(min(vall), max(vall)+1, 1) # This picks out the new range of your signal

    trimmed_signal = signal[new_range] # This gets the trimmed ekg values from signal.
    
    return trimmed_signal

"""
The following function, looks at the original signal and the trimmed signal and plots them side by side. 
    This function is more for debugging, but can be a helpful visual to see how we are trimming.
"""

def plot_og_trimmed_ekg(ptf, lead):
    
    signal = vekg.get_EKG_leads(ptf, lead)

    find_valleys(ptf, lead)

    new_signal = trim_EKG(ptf, lead)

    pyplt.figure(figsize=(10, 8))
    pyplt.subplot(2, 1, 1) # This is the first of l plots in the first column
    pyplt.plot(signal)
    pyplt.title("Original Signal")

    pyplt.subplot(2, 1, 2) # (num of plots, how many per column, cycling through the coeff.)
    pyplt.plot(new_signal)
    pyplt.title("Trimmed signal")

    pyplt.tight_layout()
    pyplt.show()

# You will want an input to be the normalization frequency.
# Big question: How do you stretch out or shrink down a signal to fit in xx bpm?

# def normalize(ptf, lead_s, bpm):

    # For now bpm is a variable to be entered in, but eventually I would like to pick a fixed bpm, but I'm not sure which is the most appropriate bpm to pick.
    # When you pick the bpm, uncomment the following piece of the code and take out the variable bpm.

    # bpm = xyz