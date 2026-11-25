from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


"""
This file performs the training and testing for the XGBoost algorithm. The input of this file is the vectorizations from the Vectorization.py file.

    The vecotrization files are saved in a folder labeled by the dimension you are embedding into using the SWE. Ex. "Vectorization_R2" is the
        folder having the vectorization of the persistent homology where the SWE is embedded in R^2.
    By default the do_xgboost cycles through 5 choices of random states to get an average accuracy.

The output of this file is the accuracy for each of the vectorizations trained using the XGBoost model. This information is written to a CSV and saved as "All_Results_XGBoost.csv".
"""

def do_xgboost(X, y, learning_rate, max_depth): 

    model = XGBClassifier(
    objective='binary:logistic',
    max_depth=max_depth,
    learning_rate=learning_rate,
    n_jobs=1
    )

    print("In function before cross_validate.")

    scores = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring={
            'auc': 'roc_auc',
            'accuracy': 'accuracy'
        },
        n_jobs=-1
    )

    print("In function after cross_validate.")

    auc_scores = scores['test_auc']
    accuracy_scores = scores['test_accuracy']

    return auc_scores, accuracy_scores


Rns = ['R2', 'R3', 'R4'] # If you have a specific R_n that you are interested in, specify that here.

Vectorizations = ['Betti_Vectorization', 'Pers_Vec_Vectorization', 'Pers_Vec_Vectorization_nooverlap', 'Pers_Vec_Vectorization_nocount', 'Pers_Vec_Vectorization_nocount_nooverlap', 'Pers_Vec_Vectorization_noarea']

ending_leads = ['_all', '_V1', '_V2', '_V3', '_V1_V2', '_V1_V3', '_V2_V3']

Ns = [1, 5, 10, 100]

# XGBoost parameters
max_depth = [3, 5, 7]
# max_depth = [3]
learning_rate = [0.05, 0.1, 0.2]
# learning_rate = [0.1]
# num = [50, 75, 100]
# num = [50]
# test_size = 0.20

l = 0
while l < len(learning_rate):
    learn_rate = learning_rate[l]

    md = 0
    while md < len(max_depth):
        max_d = max_depth[md]

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

                            csv_name = Vec_title + suffix_lead + '_' + str(n) + '_' + str(R_n) + '.csv'
                            print(n)
                            file = f'Vectorizations_10_16_26/{csv_name}'

                            print(file)

                            dataset = pd.read_csv(file, header = None)
                            dataset = dataset.drop(dataset.columns[[0]], axis = 1)
                            max_col = dataset.shape[1]-1
                            X = dataset.iloc[:, 0:max_col]
                            y = dataset.iloc[:, max_col].values

                            auc_scores, accuracy_scores = do_xgboost(X, y, learn_rate, max_d)
                            mean_auc_score = auc_scores.mean()
                            std_auc_score = auc_scores.std()
                            mean_accuracy_score = accuracy_scores.mean()
                            std_accuracy_score = accuracy_scores.std()

                            print(f'Average AUC over 5 fold cross validation for {file} with N = {n} is: {mean_auc_score}')
                            print(f'Average AUC over 5 fold cross validation for {file} with N = {n} is: {mean_accuracy_score}')

                            All_info_to_be_written.append([Vec_title + suffix_lead, R_n, mean_auc_score, std_auc_score, mean_accuracy_score, std_accuracy_score])

                            i = i+1

                    else:
                        
                        csv_name = Vec_title + suffix_lead  + '_' + str(R_n) + '.csv'

                        file = f'Vectorizations_10_16_26/{csv_name}'

                        print(file)

                        dataset = pd.read_csv(file, header = 0)
                        dataset = dataset.drop(dataset.columns[[0]], axis = 1)
                        max_col = dataset.shape[1]-1
                        X = dataset.iloc[:, 0:max_col]
                        y = dataset.iloc[:, max_col].values

                        auc_scores, accuracy_scores = do_xgboost(X, y, learn_rate, max_d)
                        mean_auc_score = auc_scores.mean()
                        std_auc_score = auc_scores.std()
                        mean_accuracy_score = accuracy_scores.mean()
                        std_accuracy_score = accuracy_scores.std()

                        print(f'Average AUC over 5 fold cross validation for {file} is: {mean_auc_score}')
                        print(f'Average AUC over 5 fold cross validation for {file} with N = {n} is: {mean_accuracy_score}')

                        All_info_to_be_written.append([Vec_title + suffix_lead, R_n, mean_auc_score, std_auc_score, mean_accuracy_score, std_accuracy_score])

                    el = el+1
                
                v = v+1
            
            r = r+1

        All_info_to_be_written = np.array(All_info_to_be_written)

        output_csv_file = f"All_Results_XGBoost_lr_{learn_rate}_md_{max_d}.csv"
        with open(output_csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Vectorization_Method_and_lead", "R_n", "Mean_AUC", "STD_AUC", "Mean_Accuracy", "STD_Accuracy"])


            i = 0
            while i < len(All_info_to_be_written):
                row = All_info_to_be_written[i]
                writer.writerow(row)
                i = i+1

        print(f'################## Saved Results to {output_csv_file} ##################')


        md = md+1

    l = l+1
