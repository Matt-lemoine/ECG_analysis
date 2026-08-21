# This is for SVM.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import KFold, cross_val_score


"""
This file performs the training and testing for the XGBoost algorithm. The input of this file is the vectorizations from the Vectorization.py file.

    The vecotrization files are saved in a folder labeled by the dimension you are embedding into using the SWE. Ex. "Vectorization_R2" is the
        folder having the vectorization of the persistent homology where the SWE is embedded in R^2.
    By default the do_svm cycles through 5 choices of random states to get an average accuracy.

The output of this file is the accuracy for each of the vectorizations trained using the XGBoost model. This information is written to a CSV and saved as "All_Results_XGBoost.csv".
"""



def do_svm(X, y, C_param):

    svm_classifier = LinearSVC(C = C_param)
    scores = cross_val_score(svm_classifier, X, y, cv = 5, verbose = 2)

    return scores.mean()



Rns = ['R2_w7', 'R3_w7', 'R4_w7'] # If you have a specific R_n that you are interested in, specify that here.

Vectorizations = ['Betti_Vectorization', 'Pers_Vec_Vectorization', 'Pers_Vec_Vectorization_nooverlap', 'Pers_Vec_Vectorization_nocount', 'Pers_Vec_Vectorization_nocount_nooverlap', 'Pers_Vec_Vectorization_noarea']

ending_leads = ['_all', '_V1', '_V2', '_V3', '_V1_V2', '_V1_V3', '_V2_V3']

Ns = [1, 5, 10, 100]

# SVM parameters
C_param = [0.01, 0.1, 0.25, 0.5]

c = 0
while c < len(C_param):
    Cparam = C_param[c]

    All_info_to_be_written = []

    r = 0
    while r < len(Rns):
        R_n = Rns[r]

        v = 0
        while v < len(Vectorizations):
            Vec_title = Vectorizations[v]

            el = 0
            while el < len(ending_leads):
                suffix_lead = ending_leads[el]

                if Vec_title == 'Betti_Vectorization':
                    i = 0
                    while i < len(Ns):
                        n = Ns[i]

                        current_vec_title = Vec_title + suffix_lead + '_' + str(n)
                        current_vec = current_vec_title + '.csv'

                        csv_name = current_vec
                        file = f'Vectorization_{R_n}_off1/{csv_name}'

                        print(file)

                        max_col = 2*n

                        dataset = pd.read_csv(file, header = None)
                        dataset = dataset.drop(dataset.columns[[0,1]], axis = 1)
                        X = dataset.iloc[:, 0:max_col]
                        y = dataset.iloc[:, max_col].values

                        avg_accuracy = do_svm(X, y, Cparam)
                        
                        print(f'Average accuracy with 5-fold cross validation for {file} with N = {n} is: {avg_accuracy}')

                        All_info_to_be_written.append([current_vec_title, R_n, avg_accuracy])

                        i = i+1

                elif Vec_title == 'Pers_Vec_Vectorization' or Vec_title == 'Pers_Vec_Vectorization_nooverlap':
                    
                    current_vec_title = Vec_title + suffix_lead
                    current_vec = current_vec_title + '.csv'

                    csv_name = current_vec
                    file = f'Vectorization_{R_n}/{csv_name}'

                    print(file)

                    col_names = ['Total 1 area', 'Total 1 Count', 'Total 0 area', 'Total 0 Count', 'Value']

                    dataset = pd.read_csv(file)
                    dataset = dataset.drop(dataset.columns[[0,1]], axis = 1)
                    X = dataset.iloc[:, 0:4]
                    y = dataset.iloc[:, 4].values

                    avg_accuracy = do_svm(X, y, Cparam)

                    print(f'Average accuracy with 5-fold cross validation for {file} is: {avg_accuracy}')

                    All_info_to_be_written.append([current_vec_title, R_n, avg_accuracy])

                elif Vec_title == 'Pers_Vec_Vectorization_nocount' or Vec_title =='Pers_Vec_Vectorization_nocount_nooverlap':

                    current_vec_title = Vec_title + suffix_lead
                    current_vec = current_vec_title + '.csv'

                    csv_name = current_vec
                    file = f'Vectorization_{R_n}/{csv_name}'

                    print(file)

                    col_names = ['Total 1 area', 'Total 0 area', 'Value']

                    dataset = pd.read_csv(file)
                    dataset = dataset[col_names]
                    X = dataset.iloc[:, 0:2] 
                    y = dataset.iloc[:, 2].values

                    avg_accuracy = do_svm(X, y, Cparam)

                    print(f'Average accuracy with 5-fold cross validation for {file} is: {avg_accuracy}')

                    All_info_to_be_written.append([current_vec_title, R_n, avg_accuracy])

                elif Vec_title == 'Pers_Vec_Vectorization_noarea':
                    
                    current_vec_title = Vec_title + suffix_lead
                    current_vec = current_vec_title + '.csv'

                    csv_name = current_vec
                    file = f'Vectorization_{R_n}/{csv_name}'

                    print(file)

                    col_names = ['Total 0 Count', 'Total 1 Count', 'Value']
                    dataset = pd.read_csv(file)
                    dataset = dataset[col_names]
                    X = dataset.iloc[:, 0:2] 
                    y = dataset.iloc[:, 2].values

                    avg_accuracy = do_svm(X, y, Cparam)

                    print(f'Average accuracy with 5-fold cross validation for {file} is: {avg_accuracy}')

                    All_info_to_be_written.append([current_vec_title, R_n, avg_accuracy])

                el = el+1
            
            v = v+1
        
        r = r+1

    All_info_to_be_written = np.array(All_info_to_be_written)

    output_csv_file = f"SVM_CSV_Results/All_Results_SVM_Cparam_{Cparam}.csv"
    with open(output_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Vectorization_Method_and_lead", "R_n", "Accuracy"])

        i = 0
        while i < len(All_info_to_be_written):
            row = All_info_to_be_written[i]
            writer.writerow(row)
            i = i+1

    print(f'################## Saved Results to {output_csv_file} ##################')

    c = c+1

