# This is for running the messy EKGs by hand.

"""
Analysis of Electrocardiography and its properties using the persistent homology of its sliding window embedding.

Created by Matthew Lemoine
February 2026
"""

"""
Notes on this file:
    In this file, we begin by taking in a list of the patient numbers and putting them in a list called "all_folder_names".
    Then the while loop cycles through all the patients to calculate the needed things.

    Our two while loops cycle through the patients, then the leads we are looking at.

    When you are importing the data, you do not need to import the data in main_EKG, because it is automatically imported
    in each of the other files, you just need to specific where those files should look.
"""

import numpy as np
import matplotlib as plt
import wfdb
import os

from pathlib import Path

import Visualize_EKG as vekg
import Normalize_EKG as nekg
import Wavelet_approx as wa
import SWE as swe
import construct_complex as cc

pat_id = input('What is the pat_id? ')

lead_in = input('What is the lead? ')

lead = [f'{lead_in}']

lead_s = ['V1', 'V2', 'V3'] # These are the ones we are interested in at first, because these are the known indicators of Brugada Syndrome.

# This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

print(f"START looking at lead {lead}")

naming_things = f'{pat_id}_{lead_in}' # This is for labeling things. It calls the patient number and lead treating it as a str.


"Step 1: Point to the data"

ptf = f'Brugada_dataset/files3(messy)/{pat_id}/{pat_id}'
lead_in_cycle = [f'{lead}'] # This records which lead we are looking at.


"Step 2: Normalize EKG signal"

# There is no option to edit this from main_EKG.py. To edit any aspect of the normalization you must go to Normalize_EKG.py and edit there.
# We have made a choice to pick out the middle-most 6 peaks from the EKG reading. This can be edited in the trim_EKG function.


"Step 4: SWE"

# tau = 0.5
tau = 1.25 # I use this one for the Cubic Spline approximation.
M = 2
# breakup_interval_more = 'insert more than len(signal) to break up your signal into more pieces and get more points in SWE.'

# projected_points = np.array(swe.SWE_get_points_nd(ptf, lead_in_cycle, wavelet, level_decomp, tau, M)) # This is using the Wavelet to approximate the EKG.
# projected_points = np.array(swe.SWE_w_spline(ptf, lead_in_cycle, tau, M)) # This is using the Cubic Spline to approximate the EKG.

pro_points = swe.SWE_no_approx_by_hand(ptf, lead, M)
projected_points = np.array(pro_points)

# if M == 1: # If you want the plot pictures uncomment this if-then loop.
#     swe.save_plot_2d(projected_points, naming_things)
# elif M == 2:
#     swe.save_plot_3d(projected_points, naming_things)

print(f"Performed Sliding Window Embedding. Shape = {projected_points.shape}. Now moving to Persistent Homology.")

# if leads_to_cycle_through == 0: # this is to double check that the leads aren't funky looking.
#     length_of_trim_V1 = len(projected_points)
# elif leads_to_cycle_through == 1:
#     length_of_trim_V2 = len(projected_points)
# elif leads_to_cycle_through == 2:
#     length_of_trim_V3 = len(projected_points)
#     if abs(length_of_trim_V1 - length_of_trim_V2) > 50 or abs(length_of_trim_V1 - length_of_trim_V3) > 50 or abs(length_of_trim_V3 - length_of_trim_V2) > 50:
#         print(f"{all_folder_names[cycle]} is a funky one. Double check the trim.")
#         funky_patients.append(all_folder_names[cycle])
#     else:
#         print('All is well')

"Step 5: Persistent Homology of SWE point cloud"

output_csv_name = f"{naming_things}_pers_info"
output_file_graph = f"{naming_things}_pers_diagram"

max_dimension = 2
max_edge_length = 1.5

persistence = cc.construct_calculate(projected_points, max_dimension, max_edge_length, output_csv_name)

try:
    cc.persistence_graph(persistence, output_file_graph)
except Exception as e:
    print(f"An error occurred during the persistence graph: {e}")

print("Persistence Calculated")
