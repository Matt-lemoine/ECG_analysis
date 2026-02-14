"""
This is Python code that starts to outline the pipeline for EKG analsis using Pers.Homol., Wavelets, and SWEs. 

Created by Matthew Lemoine
Initialization: Feb. 6, 2026
Finalization: --- --, 2026
"""

"""
Notes on this file:
    In this file, we begin by taking in a list of the patient numbers and putting them in a list called "all_folder_names".
    Then the while loop cycles through all the patients to calculate the needed things.
"""

"""
Note to self:
Ideal Pipeline:
 - The data is something (csv, png, jpeg, html, carrier pigeon, idk). (DONE)
 - We need to first get the wavelet approximation of the EKG.
 - Then hand the wavelet to SWE
 - When the wavelet has been converted into a point cloud in R^{M+1},
    persistent homol. is calcuated on this point cloud.
 - Analysis is performed on the pers. homol. of the point cloud of the wavelet approximation of the EKG.
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import wfdb

from pathlib import Path

import Visualize_EKG as vekg
import Wavelet_approx as wa
import construct_comples as cc

lead_s = ['V1', 'V2', 'V3'] # These are the ones we are interested in at first, because these are the known indicators of Brugada.

# This little piece, gives you the ability to cycle through all the folders.

get_to_files = Path('./Brugada_dataset/files')

all_folder_names = []

for subdir in get_to_files.iterdir():
    if subdir.is_dir():
        all_folder_names.append(subdir.name)

num_patients = len(all_folder_names)

cycle = 0

# This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

while cycle < num_patients:

    naming_things = f'{all_folder_names[cycle]}' # This is for labeling things. It calls the patient number treating it as a str.
    

    "Step 1: Import the data"

    ptf = f'Brugada_dataset/files/{all_folder_names[cycle]}/{all_folder_names[cycle]}' # This gets you to the patient.

    v1_v3 = vekg.get_EKG_leads(ptf, lead_s)


    "Step 2: Wavelet Approximation"

    # First pull the leads you want

    # Ideally for each of these pieces you would have a separate file that has the functions that you would import in the preamble.


    "Step 3: SWE"


    "Step 4: Persistent Homology of SWE point cloud"

    # The following three lines are for naming purposes. I don't have steps 2 and 3 done, so the naming may change.
    # output_file_barcodes = f"Persistence_Barcode_{dataset.replace(".csv",".png").replace("TDA-code-main NEW/","")}"
    # output_file_graph = f"Persistence_Diagram_{dataset.replace(".csv",".png").replace("TDA-code-main NEW/","")}"
    # output_csv_name = f"Persistence_Info_{dataset.replace("TDA-code-main NEW/","")}"

    max_dimension = 3
    max_edge_length = 3

    # Call the function with the specified parameters
    persistence = cc.construct_calculate("normalized_data", max_dimension, max_edge_length, output_csv_name)


    # Now compute the persistence barcodes or graph 
    try:
        cc.persistence_barcodes(persistence, output_file_barcodes)
    except Exception as e:
        print(f"An error occurred during the persistence barcodes: {e}")


    try:
        cc.persistence_graph(persistence, output_file_graph)
    except Exception as e:
        print(f"An error occurred during the persistence graph: {e}")


    "Step 5: Analysis of Persistent Homology"
    # this is not done on the computer, as far as I know.

    cycle = cycle + 1