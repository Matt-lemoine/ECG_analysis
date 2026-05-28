import numpy as np
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

def trim_EKG_by_hand(ptf, lead, naming_things):

    Lower_bound_trim = int(input(f"The lower bound of lead {lead} the trimmed EKG is "))

    Upper_bound_trim = int(input(f"The upper bound of lead {lead} the trimmed EKG is "))

    return [naming_things, lead, Lower_bound_trim, Upper_bound_trim]


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

def trim_EKG_is_good(ptf, lead):

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

    # if yes_no == "n":
    #     lead_s = ['V1', 'V2', 'V3']
    #     vekg.plot_ekg(ptf, lead_s, ptf)

    #     [naming_things, lead, Lower_bound_trim, Upper_bound_trim] = trim_EKG_by_hand(ptf, lead, ptf)

    #     new_range = np.arange(Lower_bound_trim, Upper_bound_trim+1, 1)

    #     trimmed_signal = signal[new_range]

    # else:
    #     trimmed_signal = trimmed_signal
    
    # return trimmed_signal


# You will want an input to be the normalization frequency.
# Big question: How do you stretch out or shrink down a signal to fit in xx bpm?

# def normalize(ptf, lead_s, bpm):

    # For now bpm is a variable to be entered in, but eventually I would like to pick a fixed bpm, but I'm not sure which is the most appropriate bpm to pick.
    # When you pick the bpm, uncomment the following piece of the code and take out the variable bpm.

    # bpm = xyz


# """
# The following functions are to handle the EKGs that have at least 1 lead that is correct and do not need to be
# hand-trimmed.
# """

# def trim_EKG_using_good_lead(ptf, good_lead, bad_lead):

#     num_peaks = 6  # To edit the normalized num_peaks, edit here.

#     if len(good_lead) == 1:
#         signal = vekg.get_EKG_leads(ptf, good_lead)
#     else:
#         error = 'You can only import one lead at a time for trim_EKG.'
#         return print(error)

#     valleys = find_valleys(ptf, good_lead)
#     num_valleys = len(valleys)

#     if num_valleys < num_peaks + 2:
#         error = f"You need at least two more peaks in your reading than your valleys in order to trim, you have {num_valleys} valleys"
#         return print(error)
#     else:
#         diff = num_valleys - num_peaks
#         front_trim = diff // 2
#         vall = valleys[front_trim - 1 : front_trim + num_peaks + 1]

#     new_range = np.arange(min(vall), max(vall)+1, 1) # This picks out the new range of your signal

#     bad_signal = vekg.get_EKG_leads(ptf, bad_lead)

#     trimmed_signal = bad_signal[new_range] # So we use the new_range of the good_lead to get the trimmed signal for the bad_lead.
    
#     return trimmed_signal

# def trim_EKG_by_hand(ptf, lead):

#     if len(lead) == 1:
#         signal = vekg.get_EKG_leads(ptf, lead)
#     else:
#         error = 'You can only import one lead at a time for trim_EKG.'
#         return print(error)

#     vekg.plot_ekg(ptf, lead, ptf)

#     Lower_bound_trim = int(input("The lower bound of the trimmed EKG is "))

#     vekg.plot_ekg(ptf, lead, ptf)

#     Upper_bound_trim = int(input("The upper bound of the trimmed EKG is "))

#     naming_things = int(naming_things)

#     info = [Lower_bound_trim, Upper_bound_trim]

#     new_range = np.arange(Lower_bound_trim, Upper_bound_trim+1 , 1) # This picks out the new range of your signal

#     trimmed_signal = signal[new_range] # This gets the trimmed ekg values from signal.
    
#     return trimmed_signal

# pat_id = 1358245
# ptf = f"Brugada_dataset/files/{pat_id}/{pat_id}"
# lead_s = ['V1', 'V2', 'V3']

# i = 0
# while i < len(lead_s):
#     plot_og_trimmed_ekg(ptf, [f'{lead_s[i]}'])
#     i = i+1

# from pathlib import Path

# lead_s = ['V1', 'V2', 'V3'] # These are the ones we are interested in at first, because these are the known indicators of Brugada Syndrome.

# get_to_files = Path('Brugada_dataset/files')

# all_folder_names = []

# for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient numbers)
#     if subdir.is_dir():
#         all_folder_names.append(subdir.name)

# num_patients = len(all_folder_names)

# i = 0
# pat_id_lb_ub = []
# while i < 2:

#     patient_id = all_folder_names[i]

#     ptf = f'Brugada_dataset/files/{patient_id}/{patient_id}'

#     row_to_add = trim_EKG_by_hand(ptf, lead_s, patient_id)

#     pat_id_lb_ub.append(row_to_add)
    
#     i = i+1

# output_csv_file = "trimming_EKGs.csv"
# with open(output_csv_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Patient_id", "Lower_bound", "Upper_bound"])
#     for [pi, lb, ub] in pat_id_lb_ub:
#         writer.writerow([pi, lb, ub])
    