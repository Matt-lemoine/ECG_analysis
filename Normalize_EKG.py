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

    The following function finds all the places in the single lead we are looking at, where the mV dips
    near -1. This happens when the heart is re-regulating itself after a contraction of the cardiac muscles. (I need to research what is happening here and why it goes to -1.)

    When we find the valleys of our EKG, then we can trim the EKG to have only a few of the peaks, and normalize the EKG.





     ISSUE::: There is an issue with a possible misreading where the valley picks up slightly for one centisecond, and causes a valley in the valley. I need to do something that says if two entries are within 50 or so, throw one of them out. 





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
In the following code, we want to trim our EKG to a standard length so that heartbeat does effect our comparison.
    We make a choice here to trim our EKG to the middle-most complete 8 peaks <--> we need to keep the middle-most 10 valleys.

    So for example, if I want the middle most 8 complete peaks, then I need to keep one extra valley before and one valley after
    the 8 peaks that I care about. So I need to keep track of the middle most 10 valleys.

    Maybe I should do the middle most 6 peaks, and keep the middle most 8 valleys? I could run it with a few different options of num_valleys and then pick on that keeps everything nice and neat.

    Notes on the if-else piece after 'num_valleys = len(valleys)':
      You have to check that you have enough space to get the num_peaks you're looking for, so if num_valleys isn't small enough there's an error.
      Then when you have the right num_peaks, you find how much you want to trim off the front and back of the signal by finding diff.
      They we make an assumption that we want the middle-most part of the signal, and we trim more off the back if there is not an even number
      of peaks and valleys. And vall is the trimmed signal.

trim_EKG only works for V1-V3. I need to adjust it to work for V4-V6.
"""

def trim_EKG(ptf, lead, num_peaks):

    # For now num_peaks is a variable to be entered in, but eventually I would like to pick a fixed num_peaks, but I'm not sure which is the most appropriate num_peaks to pick.
    # When you pick the num_peaks, uncomment the following piece of the code and take out the variable num_peaks.

    # num_peaks = xyz

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

    new_range = np.arange(min(vall), max(vall)+1, 1) # This tells you the range of the trimmed ekg.

    trimmed_signal = signal[new_range] # This gets the trimmed ekg values from signal.
    
    return trimmed_signal

def plot_og_trimmed_ekg(ptf, lead, num_peaks):
    
    signal = vekg.get_EKG_leads(ptf, lead)

    find_valleys(ptf, lead)

    new_signal = trim_EKG(ptf, lead, num_peaks)

    pyplt.figure(figsize=(10, 8))
    pyplt.subplot(2, 1, 1) # This is the first of l plots in the first column
    pyplt.plot(signal)
    pyplt.title("Original Signal")

    pyplt.subplot(2, 1, 2) # (num of plots, how many per column, cycling through the coeff.)
    pyplt.plot(new_signal)
    pyplt.title("Trimmed signal")

    pyplt.tight_layout()
    pyplt.show()

# def normalize(ptf, lead_s, bpm):

    # For now bpm is a variable to be entered in, but eventually I would like to pick a fixed bpm, but I'm not sure which is the most appropriate bpm to pick.
    # When you pick the bpm, uncomment the following piece of the code and take out the variable bpm.

    # bpm = xyz