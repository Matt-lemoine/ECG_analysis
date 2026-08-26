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

    In this file, the output is a CSV and a PNG for each patient and lead detailing the Persistent homology. The CSV file will be used later to train the XGBoost.
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import wfdb
import os
import os.path

from pathlib import Path

import Visualize_EKG as vekg
import Normalize_EKG as nekg
import SWE as swe
import construct_complex as cc

lead_s = ['V1', 'V2', 'V3'] # These are the ones we are interested in at first, because these are the known indicators of Brugada Syndrome.

get_to_files = Path(f'./Brugada_dataset/files') # This tells python where to look for the names of the Patient IDs.
all_folder_names = []
for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient IDs.)
    if subdir.is_dir():
        all_folder_names.append(subdir.name)

num_patients = len(all_folder_names)


cycle = 0
while cycle < num_patients: # Cycles through patients.

    print(f"****** START ****** looking at patient {all_folder_names[cycle]}")

    leads_to_cycle_through = 0
    while leads_to_cycle_through < len(lead_s): # Cycles through leads.

        print(f"START looking at lead {lead_s[leads_to_cycle_through]}")

        naming_things = f'{all_folder_names[cycle]}_{lead_s[leads_to_cycle_through]}' # This is for labeling things. It calls the patient number and lead treating it as a str.


        "Step 1: Point to the data"

        ptf = f'Brugada_dataset/files/{all_folder_names[cycle]}/{all_folder_names[cycle]}'
        lead_in_cycle = [f'{lead_s[leads_to_cycle_through]}'] # This records which lead we are looking at.

        
        "Step 2: Normalize EKG signal"

        # In this normalize_EKG file, it trims the EKG, then approximates the trimmed EKG with Splines, then passes to the SWE.
        
        "Step 3: SWE"

        M = 1
        while M < 4:

            pro_points = swe.SWE_w_Splines(ptf, lead_in_cycle, M)
            projected_points = np.array(pro_points)

            print(f"Performed Sliding Window Embedding. Shape = {projected_points.shape}. Now moving to Persistent Homology.")

            if len(projected_points) != 615:
                "There is a huge error there should only be 615 points."
                break

            "Step 4: Persistent Homology of SWE point cloud"

            folder = f"Persistent_Homology_R{M+1}"
            output_csv_name = f"{naming_things}_pers_info"
            output_file_graph = f"{folder}/{naming_things}_pers_diagram"

            max_dimension = 2           # This means we will look for the persistent homology for dimensions 0 and 1.
            max_edge_length = 6

            persistence = cc.construct_calculate(projected_points, max_dimension, max_edge_length, output_csv_name, folder)

            try:
                cc.persistence_graph(persistence, output_file_graph)
            except Exception as e:
                print(f"An error occurred during the persistence graph: {e}")

            print("Persistence Calculated")

            M = M+1

        if M != 4:
            break

        print(f"END looking at lead {lead_s[leads_to_cycle_through]}")

        leads_to_cycle_through += 1

    if leads_to_cycle_through!=3:
        break

    time_left = (num_patients - cycle -1)*7.5
    print(f"****** END ****** looking at patient {all_folder_names[cycle]}. Patient {cycle + 1}/{num_patients}. Approximately {time_left} minutes left.")

    cycle += 1

if cycle != 363:

    "There was a big mistake somewhere. oops"
