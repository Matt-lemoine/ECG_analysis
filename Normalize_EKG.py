import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplt
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


def plot_og_trimmed_ekg(ptf, lead):
    
    signal = vekg.get_EKG_leads(ptf, lead)

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

# def trim_EKG_by_hand(ptf, lead, naming_things):

#     Lower_bound_trim = int(input(f"The lower bound of lead {lead} the trimmed EKG is "))

#     Upper_bound_trim = int(input(f"The upper bound of lead {lead} the trimmed EKG is "))

#     return [naming_things, lead, Lower_bound_trim, Upper_bound_trim]


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

    pyplt.clf()
    pyplt.subplot(2, 1, 1) # This is the first of l plots in the first column
    pyplt.plot(signal)
    pyplt.title(f"Original Signal {ptf}")

    pyplt.subplot(2, 1, 2)
    pyplt.plot(trimmed_signal)
    pyplt.title("Trimmed signal")

    pyplt.tight_layout()
    pyplt.savefig("Test_Trim.png")

    yes_no = input("Does Test_Trim.png look correct? (y/n): ")

    if yes_no == "n":
        print("It doesn't look correct.")
        return "funky"
    else:
        print("It looks correct.")
        return trimmed_signal

# def trim_EKG_is_good(ptf, lead):

#     num_peaks = 6  # To edit the normalized num_peaks, edit here.

#     if len(lead) == 1:
#         signal = vekg.get_EKG_leads(ptf, lead)
#     else:
#         error = 'You can only import one lead at a time for trim_EKG.'
#         return print(error)

#     valleys = find_valleys(ptf, lead)
#     num_valleys = len(valleys)

#     if num_valleys < num_peaks + 2:
#         error = f"You need at least two more peaks in your reading than your valleys in order to trim, you have {num_valleys} valleys"
#         return print(error)
#     else:
#         diff = num_valleys - num_peaks
#         front_trim = diff // 2
#         vall = valleys[front_trim - 1 : front_trim + num_peaks + 1]

#     new_range = np.arange(min(vall), max(vall)+1, 1) # This picks out the new range of your signal

#     trimmed_signal = signal[new_range] # This gets the trimmed ekg values from signal.

#     return trimmed_signal

def trim_by_CSV(ptf, lead):

    if len(lead) == 1:
        signal = vekg.get_EKG_leads(ptf, lead)
    else:
        error = 'You can only import one lead at a time for trim_EKG.'
        return print(error)

    lb, ub = find_bounds(ptf, lead)

    new_range = np.arange(lb, ub+1, 1) # This picks out the new range of your signal

    trimmed_signal = signal[new_range] # This gets the trimmed ekg values from signal.
        
    return trimmed_signal


def trim_by_hand(ptf, lead):

    lb = int(input("What is the lower bound for your trim? "))
    ub = int(input("What is the upper bound for your trim? "))

    if len(lead) == 1:
        signal = vekg.get_EKG_leads(ptf, lead)
    else:
        error = 'You can only import one lead at a time for trim_EKG.'
        return print(error)

    new_range = np.arange(lb, ub+1, 1) # This picks out the new range of your signal

    trimmed_signal = signal[new_range] # This gets the trimmed ekg values from signal.

    pyplt.clf()
    pyplt.subplot(2, 1, 1) # This is the first of l plots in the first column
    pyplt.plot(signal)
    pyplt.title(f"Original Signal {ptf}")

    pyplt.subplot(2, 1, 2)
    pyplt.plot(trimmed_signal)
    pyplt.title("Trimmed signal")

    pyplt.tight_layout()
    pyplt.savefig("Test_Trim.png")

    yes_no = input("Does Test_Trim.png look correct? (y/n): ")

    if yes_no == "n":
        print("If it doesn't look correct, you messed up the trimming.")
    else:
        print("It looks correct.")
        
    return trimmed_signal

def trim_EKG_bounds(ptf, lead): # This code was used to trim the ekg's by hand using images and pyplots of them. It uses the find_valleys to get a good start.

    def trim_by_hand_in_trim_EKG(ptf, lead):

        lb = int(input("What is the lower bound for your trim? "))
        ub = int(input("What is the upper bound for your trim? "))

        if len(lead) == 1:
            signal = vekg.get_EKG_leads(ptf, lead)
        else:
            error = 'You can only import one lead at a time for trim_EKG.'
            return print(error)

        new_range = np.arange(lb, ub+1, 1) # This picks out the new range of your signal

        lb_ub = np.array([lb, ub+1])

        trimmed_signal = signal[new_range] # This gets the trimmed ekg values from signal.

        pyplt.clf()
        pyplt.subplot(2, 1, 1) # This is the first of l plots in the first column
        pyplt.plot(signal)
        pyplt.title(f"Original Signal {ptf}")

        pyplt.subplot(2, 1, 2)
        pyplt.plot(trimmed_signal)
        pyplt.title("Trimmed signal")

        pyplt.tight_layout()
        pyplt.savefig("Test_Trim.png")

        yes_no = input("Does Test_Trim.png look correct? (y/n): ")

        if yes_no == "n":
            print("If it doesn't look correct, you messed up the trimming.")
        else:
            print("It looks correct.")
            
        return lb_ub

    num_peaks = 6  # To edit the normalized num_peaks, edit here.

    if len(lead) == 1:
        signal = vekg.get_EKG_leads(ptf, lead)
    else:
        error = 'You can only import one lead at a time for trim_EKG.'
        return print(error)

    valleys = find_valleys(ptf, lead)
    num_valleys = len(valleys)

    if num_valleys < num_peaks + 2:
        naming_things = f'{ptf}_{lead}'
        vekg.plot_ekg(ptf, lead, naming_things)

        lb_ub = trim_by_hand_in_trim_EKG(ptf, lead)
        return lb_ub
    else:
        diff = num_valleys - num_peaks
        front_trim = diff // 2
        vall = valleys[front_trim - 1 : front_trim + num_peaks + 1]

    new_range = np.arange(min(vall), max(vall)+1, 1) # This picks out the new range of your signal

    lb_ub = np.array([min(vall), max(vall)+1])

    trimmed_signal = signal[new_range] # This gets the trimmed ekg values from signal.

    pyplt.clf()
    pyplt.subplot(2, 1, 1) # This is the first of l plots in the first column
    pyplt.plot(signal)
    pyplt.title(f"Original Signal {ptf}")

    pyplt.subplot(2, 1, 2)
    pyplt.plot(trimmed_signal)
    pyplt.title("Trimmed signal")

    pyplt.tight_layout()
    pyplt.savefig("Test_Trim.png")

    yes_no = input("Does Test_Trim.png look correct? (y/n): ")

    if yes_no == "n":

        naming_things = f'{ptf}_{lead}'
        vekg.plot_ekg(ptf, lead, naming_things)

        lb_ub = trim_by_hand_in_trim_EKG(ptf, lead)
        return lb_ub        
    else:
        print("It looks correct.")
        return lb_ub

def find_bounds(ptf, lead):

    trimming_info = pd.read_csv("All_Trimming_info.csv")
    trimming_info = np.array(trimming_info)

    pat_id = ptf.split("/",3)[-1]

    index = trimming_info[:,0] == float(pat_id)
    matching_rows = trimming_info[index]

    index = matching_rows[:,1] == lead
    matching_rows = matching_rows[index]

    lb = matching_rows[0,2]
    ub = matching_rows[0,3]

    return lb, ub