#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:17:59 2024

@author: meganfairchild

This is the main file to compute TDA for our data set.
"""

#necessary preamble
import io
import pandas as pd
import numpy as np
data_type = np.sctypeDict['float']
import gudhi

#accompanying python scripts
import construct_complex as cc
import construct_mapper as cmapper 

"""
Step 1: Import the CSV of the data set
"""

# dataset = 'TDA-code-main NEW/PD_Original_2_no3.csv'  # whichever we are currently analyzing. make sure it is in the same folder as the script. 
# #now name the output files 
# naming = f"{dataset.replace(".csv","").replace("TDA-code-main NEW/","")}"
# output_file_barcodes = f"Persistence_Barcode_{dataset.replace(".csv",".png").replace("TDA-code-main NEW/","")}"
# output_file_graph = f"Persistence_Diagram_{dataset.replace(".csv",".png").replace("TDA-code-main NEW/","")}"
# output_csv_name = f"Persistence_Info_{dataset.replace("TDA-code-main NEW/","")}"

# If you have a lot of different types of MAPPER or Persistence you're running, then uncomment the while loop. And indent where needed.

i = 0

# files = ['TDA-code-main NEW/PD_Original_2.csv','TDA-code-main NEW/PD_Original_2_no3.csv','TDA-code-main NEW/PD_Original_2_noALM.csv','TDA-code-main NEW/PD_Original_2_noBFP.csv','TDA-code-main NEW/PD_Original_2_noBMD.csv','TDA-code-main NEW/PD_Original_2_onlyALM.csv','TDA-code-main NEW/PD_Original_2_onlyBFP.csv','TDA-code-main NEW/PD_Original_2_onlyBMD.csv']

files = ['TDA-code-main NEW/PD_Original_2_noAGE.csv']

while i < len(files):
    dataset = files[i]
    naming = dataset.replace(".csv","").replace("TDA-code-main NEW/PD_Original_2","")

    """
    Step 2: we will ensure the dataset is what we are expecting. 
    """
    data = pd.read_csv(dataset)
    def load_and_validate_csv(filepath):
        try:
            df = pd.read_csv(filepath)
            if not np.issubdtype(df.to_numpy().dtype, np.number):
                raise ValueError("The CSV file contains non-numerical values.")
            data_array = df.to_numpy()
            if data_array.size == 0:
                raise ValueError("The CSV file is empty.")
            return data_array
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            raise
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty.")
            raise
        except ValueError as e:
            print(f"Error: {e}")
            raise

    if __name__ == "__main__": #if __name__ == "__main__": checks if the script is being run directly. If true, the code block following it will execute.
        filepath=dataset
        try:
            data_array = load_and_validate_csv(filepath)
            print("Data loaded successfully.")
            print(f"Data shape: {data_array.shape}")
            # Call other scripts or functions here
        except Exception as e:
            print(f"Failed to load and validate data: {e}")
        

    """
    Step 3: Convert to a numpy array and normalize the data
    """
    data_array = data.to_numpy() 

    def z_score_normalize(data): #the function to normalize the data
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0)
        normalized_data = (data - mean_val) / std_val
        return normalized_data

    normalized_data = z_score_normalize(data_array)
    # """
    # Step 4: Next hand the numpy array to GUDHI. Compute a Rips complex.
    # (change and edit the parameters for computing persistent homology here)
    # """

    # output_file_barcodes = f"Persistence_Barcode_{dataset.replace(".csv",".png").replace("TDA-code-main NEW/","")}"
    # output_file_graph = f"Persistence_Diagram_{dataset.replace(".csv",".png").replace("TDA-code-main NEW/","")}"
    # output_csv_name = f"Persistence_Info_{dataset.replace("TDA-code-main NEW/","")}"

    # max_dimension = 3
    # max_edge_length = 4.5
    # # Call the function with the specified parameters
    # persistence = cc.construct_calculate(normalized_data, max_dimension, max_edge_length, output_csv_name)


    # # Now compute the persistence barcodes or graph 

    # try:
    #     cc.persistence_barcodes(persistence, output_file_barcodes)
    # except Exception as e:
    #     print(f"An error occurred during the persistence barcodes: {e}")


    # try:
    #     cc.persistence_graph(persistence, output_file_graph)
    # except Exception as e:
    #     print(f"An error occurred during the persistence graph: {e}")   

    # """
    # Step 5: MAPPER Time
    # If you want to run a lot of different intervals and overlaps and files, you can uncomment the following code and
    #     indent where needed throughout the whole code. (Note: The 'Intervals = [5,10,15]' is where your indentations should be.)
    # """

    Intervals = [5, 10, 15]
    overlaps = [0.75]
            
    j = 0

    while j < len(Intervals):
        k = 0
        num_intervals = Intervals[j]
        while k < len(overlaps):

            overlap_frac = overlaps[k]

            mapper_name = f"MAPPER_{naming}_{num_intervals}_intervals_{overlap_frac}_overlap"

    # num_intervals = 10
    # overlap_frac = 0.3
    # mapper_name = f"MAPPER_{naming}_{num_intervals}_intervals_{overlap_frac}_overlap"
            # try:
            #     cmapper.construct_mapper_graph_3D(normalized_data, num_intervals, overlap_frac,mapper_name)
            # except Exception as e:
            #     print(f"An error occurred during the persistence graph: {e}")  
        
            try:
                cmapper.construct_mapper_graph_2D(normalized_data, num_intervals, overlap_frac,mapper_name)
            except Exception as e:
                print(f"An error occurred during the persistence graph: {e}")  

    #  These tell your while loops to cycle through to cover all files, intervals, and overlaps.
            k = k+1
        j = j+1
    i = i+1


"""
end of script
"""