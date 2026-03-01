import numpy as np
import matplotlib as plt
import wfdb
import csv

import Visualize_EKG as vekg

def find_valleys(ptf, lead):

    if len(lead) == 1:
        signal = vekg.get_EKG_leads(ptf, lead)
    else:
        error = 'You can only import one lead at a time for find_valleys.'
        return print(error)

    test_valleys = []
    i = 0
    while i < len(signal): # This loop finds all the places where you have a local minimum.
        if i+1 == len(signal):
            break
        elif signal[i] <= signal[i+1] and signal[i] <= signal[i-1] and signal[i]<=0:
            test_valleys.append(i)
            i += 1
        else:
            i +=1
    
    # test_valleys is every local minimum in your EKG that is less than 0.

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
    # The 15 here is arbitrary, it corresponds to 15 centiseconds in the ekg reading. It does not make sense that 
    #   there would be 2 valleys withing 15 centiseconds of each other.

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


def trim_EKG(ptf, lead):

    num_peaks = 6  # To edit the normalized num_peaks, edit here.

    if len(lead) == 1:
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


def plot_og_trimmed_ekg(ptf, lead):
    
    signal = vekg.get_EKG_leads(ptf, lead)

    find_valleys(ptf, lead)

    new_signal = trim_EKG(ptf, lead)

    pyplt.figure(figsize=(10, 8))
    pyplt.subplot(2, 1, 1) # This is the first of l plots in the first column
    pyplt.plot(signal)
    pyplt.title("Original Signal")

    pyplt.subplot(2, 1, 2)
    pyplt.plot(new_signal)
    pyplt.title("Trimmed signal")

    pyplt.tight_layout()
    return pyplt.show()


# You will want an input to be the normalization frequency.
# Big question: How do you stretch out or shrink down a signal to fit in xx bpm?

# def normalize(ptf, lead_s, bpm):

    # For now bpm is a variable to be entered in, but eventually I would like to pick a fixed bpm, but I'm not sure which is the most appropriate bpm to pick.
    # When you pick the bpm, uncomment the following piece of the code and take out the variable bpm.

    # bpm = xyz