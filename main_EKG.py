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

lead_s = ['V1', 'V2', 'V3'] # These are the ones we are interested in at first, because these are the known indicators of Brugada Syndrome.

subset = 6 # Change this when you move from one subset set to the next. (Subsets completed: 1, 2, 3, 4, 5, )

# get_to_files = Path(f'./Subset_{subset}/files')

get_to_files = Path('./Subsets_(done)/Brugada_subset/files') # This is for testing.

all_folder_names = []

for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient numbers)
    if subdir.is_dir():
        all_folder_names.append(subdir.name)

num_patients = len(all_folder_names)

# This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

cycle = 0
while cycle < num_patients: # Cycles through patients.

    print(f"****** START ****** looking at patient {all_folder_names[cycle]}")

    leads_to_cycle_through = 0
    while leads_to_cycle_through < len(lead_s): # Cycles through leads.

        print(f"START looking at lead {lead_s[leads_to_cycle_through]}")

        naming_things = f'{all_folder_names[cycle]}_{lead_s[leads_to_cycle_through]}' # This is for labeling things. It calls the patient number and lead treating it as a str.


        "Step 1: Point to the data"

        # ptf = f'Subset_{subset}/files/{all_folder_names[cycle]}/{all_folder_names[cycle]}' # This gets you to the patient.

        ptf = f'Subsets_(done)/Brugada_subset/files/{all_folder_names[cycle]}/{all_folder_names[cycle]}'

        lead_in_cycle = [f'{lead_s[leads_to_cycle_through]}'] # This records which lead we are looking at.

        
        "Step 2: Normalize EKG signal"

        # There is no option to edit this from main_EKG.py. To edit any aspect of the normalization you must go to Normalize_EKG.py and edit there.
        # We have made a choice to pick out the middle-most 6 peaks from the EKG reading. This can be edited in the trim_EKG function.

        
        "Step 3: Wavelet Approximation"

        # I have been using 'db3' and decomp = 4. 
        # There is no option to edit more from main_EKG.py. To edit more aspects of the wavelet approximation you must go to wavelet_approx.py and edit there.

        wavelet = 'db3'
        level_decomp = 4

        
        "Step 4: SWE"

        # tau = 0.5
        tau = 1.25 # I use this one for the Cubic Spline approximation.
        M = 2
        # breakup_interval_more = 'insert more than len(signal) to break up your signal into more pieces and get more points in SWE.'

        # projected_points = np.array(swe.SWE_get_points_nd(ptf, lead_in_cycle, wavelet, level_decomp, tau, M)) # This is using the Wavelet to approximate the EKG.
        projected_points = np.array(swe.SWE_w_spline(ptf, lead_in_cycle, tau, M)) # This is using the Cubic Spline to approximate the EKG.

        # if M == 1: # If you want the plot pictures uncomment this if-then loop.
        #     swe.save_plot_2d(projected_points, naming_things)
        # elif M == 2:
        #     swe.save_plot_3d(projected_points, naming_things)

        print(f"Performed Sliding Window Embedding. Shape = {projected_points.shape}. Now moving to Persistent Homology.")


        "Step 5: Persistent Homology of SWE point cloud"

        output_csv_name = f"{naming_things}_pers_info"
        output_file_graph = f"{naming_things}_pers_diagram"

        max_dimension = 2
        max_edge_length = 1

        persistence = cc.construct_calculate(projected_points, max_dimension, max_edge_length, output_csv_name)

        try:
            cc.persistence_graph(persistence, output_file_graph)
        except Exception as e:
            print(f"An error occurred during the persistence graph: {e}")

        print("Persistence Calculated")

        print(f"END looking at lead {lead_s[leads_to_cycle_through]}")

        leads_to_cycle_through += 1

    time_left = (num_patients - cycle -1)*2.5
    print(f"****** END ****** looking at patient {all_folder_names[cycle]}. Patient {cycle + 1}/{num_patients}. About {time_left} minutes left.")

    cycle += 1