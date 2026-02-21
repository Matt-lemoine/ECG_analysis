"""
Analysis of Electrocardiography and its properties using the persistent homology of its sliding window embedding.

Created by Matthew Lemoine
Initialization: Feb. 6, 2026
Finalization: Feb. , 2026
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
import pandas as pd
import matplotlib as plt
import wfdb

from pathlib import Path

import Visualize_EKG as vekg
import Wavelet_approx as wa
import SWE as swe
import construct_complex as cc

lead_s = ['V1', 'V2', 'V3'] # These are the ones we are interested in at first, because these are the known indicators of Brugada.

# This little piece, gives you the ability to cycle through all the folders.

# get_to_files = Path('./Brugada_dataset/files') ## UNCOMMENT THIS LATER

get_to_files = Path('./Brugada_subset/files') ## This is for TESTING (REMOVE).

all_folder_names = []

for subdir in get_to_files.iterdir():
    if subdir.is_dir():
        all_folder_names.append(subdir.name)

# num_patients = len(all_folder_names) ## UNCOMMENT THIS LATER

num_patients = 1 ## This is for TESTING (REMOVE). THIS WILL ONLY DO THE FIRST DUDE IN THE FOLDERS.


cycle = 0

# This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

while cycle < num_patients:

    print(f"Currently looking at Patient {all_folder_names[cycle]}.")

    naming_things = f'{all_folder_names[cycle]}' # This is for labeling things. It calls the patient number treating it as a str.
    
    # All subsequent steps must be done one lead at a time, so we cycle through the specified leads.

    leads_to_cycle_through = 0

    while leads_to_cycle_through < len(lead_s):

        print(f"Currently looking at lead {lead_s[leads_to_cycle_through]}.")


        "Step 1: Point to the data"

        # ptf = f'Brugada_dataset/files/{all_folder_names[cycle]}/{all_folder_names[cycle]}' # This gets you to the patient. ## UNCOMMENT THIS LATER

        ptf = f'Brugada_subset/files/{all_folder_names[cycle]}/{all_folder_names[cycle]}' # This gets you to the patient. TESTING(REMOVE)

        lead_in_cycle = [f'{lead_s[leads_to_cycle_through]}']
        
        
        "Step 2: Wavelet Approximation"

        # The wavelet I've been using is db3 with a level_decomp of 4.

        wavelet = 'db3'
        level_decomp = 4

        # The Wavelet function approximation at time t is called by the following function:
            # wa. wavelet(ptf, lead_s, wavelet, level_decomp, t)
            # This function is built into the SWE file, but you still need to define the wavelet, and level_decomp.

        
        "Step 3: SWE"



        """"I need to double check the len(signal) piece in the SWE_get_points_nd function, becuase right now it is breaking up your signal in one pieces."""



        tau = 0.5
        M = 2

        projected_points = swe.SWE_get_points_nd_TESTING(ptf, lead_in_cycle, wavelet, level_decomp, tau, M, 300)  ## Remove the TESTING and the 300 when you get this part done.

        projected_points = np.array(projected_points)

        print(f"Performed Sliding Window Embedding. Shape = {projected_points.shape}. Now moving to Persistent Homology.")

        
        "Step 4: Persistent Homology of SWE point cloud"

        output_csv_name = f"{naming_things}_pers_info"
        output_file_graph = f"{naming_things}_pers_diagram"

        max_dimension = M
        max_edge_length = 1

        # Call the function with the specified parameters
        persistence = cc.construct_calculate(projected_points, max_dimension, max_edge_length, output_csv_name)

        try:
            cc.persistence_graph(persistence, output_file_graph)
        except Exception as e:
            print(f"An error occurred during the persistence graph: {e}")

        print("Persistence Calculated.")

        leads_to_cycle_through += 1


    "Step 5: Analysis of Persistent Homology"
    # this is not done on the computer, as far as I know.

    cycle = cycle + 1