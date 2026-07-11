from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
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



def do_xgboost(X, y, max_depth, learning_rate, num, test_size):

    to_avg = []
    random_state = 0
    while random_state < 5:

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size,  random_state=random_state)

        xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

        params = {
            'objective': 'binary:logistic',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
        }
        num = num
        model = xgb.train(params=params,dtrain=xgb_train,num_boost_round=num)

        preds = model.predict(xgb_test)
        preds = np.round(preds)

        accuracy= accuracy_score(y_test, preds)

        to_avg.append(accuracy)

        random_state = random_state+1
    
    avg_accuracy = (sum(to_avg)/len(to_avg))*100

    return avg_accuracy





Rns = ['R2', 'R3', 'R4'] # If you have a specific R_n that you are interested in, specify that here.

Vectorizations = ['Betti_Vectorization', 'Pers_Vec_Vectorization', 'Pers_Vec_Vectorization_nooverlap', 'Pers_Vec_Vectorization_nocount', 'Pers_Vec_Vectorization_nocount_nooverlap', 'Pers_Vec_Vectorization_noarea']

ending_leads = ['_all', '_V1', '_V2', '_V3', '_V1_V2', '_V1_V3', '_V2_V3']

Ns = [1, 5, 10, 100]

# XGBoost parameters
max_depths = 3
learning_rates = 0.1
num = 50
test_size = 0.20
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

                    current_vec_title = Vec_title + suffix_lead + '_' + str(n) + f"_{R_n}"
                    current_vec = current_vec_title + '.csv'

                    csv_name = current_vec
                    file = csv_name

                    print(file)

                    max_col = 2*n

                    dataset = pd.read_csv(file, header = None)
                    dataset = dataset.drop(dataset.columns[[0,1]], axis = 1)
                    X = dataset.iloc[:, 0:max_col]
                    y = dataset.iloc[:, max_col].values

                    avg_accuracy = do_xgboost(X, y, max_depth, learning_rate, num, test_size)
                    
                    print(f'Average accuracy over 5 random_states for {file} with N = {n} is: {avg_accuracy}')

                    All_info_to_be_written.append([current_vec_title, R_n, avg_accuracy])

                    i = i+1

            elif Vec_title == 'Pers_Vec_Vectorization' or Vec_title == 'Pers_Vec_Vectorization_nooverlap':
                
                current_vec_title = Vec_title + suffix_lead + f"_{R_n}"
                current_vec = current_vec_title + '.csv'

                csv_name = current_vec
                file = csv_name

                print(file)

                col_names = ['Total 1 area', 'Total 1 Count', 'Total 0 area', 'Total 0 Count', 'Value']

                dataset = pd.read_csv(file)
                dataset = dataset[col_names]
                X = dataset.iloc[:, 0:4]
                y = dataset.iloc[:, 4].values

                avg_accuracy = do_xgboost(X, y, max_depth, learning_rate, num, test_size)

                print(f'Average accuracy over 5 random_states for {file} is: {avg_accuracy}')

                All_info_to_be_written.append([current_vec_title, R_n, avg_accuracy])

            elif Vec_title == 'Pers_Vec_Vectorization_nocount' or Vec_title =='Pers_Vec_Vectorization_nocount_nooverlap':

                current_vec_title = Vec_title + suffix_lead + f"_{R_n}"
                current_vec = current_vec_title + '.csv'

                csv_name = current_vec
                file = csv_name

                print(file)

                col_names = ['Total 1 area', 'Total 0 area', 'Value']

                dataset = pd.read_csv(file)
                dataset = dataset[col_names]
                X = dataset.iloc[:, 0:2] 
                y = dataset.iloc[:, 2].values

                avg_accuracy = do_xgboost(X, y, max_depth, learning_rate, num, test_size)

                print(f'Average accuracy over 5 random_states for {file} is: {avg_accuracy}')

                All_info_to_be_written.append([current_vec_title, R_n, avg_accuracy])

            elif Vec_title == 'Pers_Vec_Vectorization_noarea':

                current_vec_title = Vec_title + suffix_lead + f"_{R_n}"
                current_vec = current_vec_title + '.csv'

                csv_name = current_vec
                file = csv_name

                print(file)

                col_names = ['Total 0 Count', 'Total 1 Count', 'Value']
                dataset = pd.read_csv(file)
                dataset = dataset[col_names]
                X = dataset.iloc[:, 0:2] 
                y = dataset.iloc[:, 2].values

                avg_accuracy = do_xgboost(X, y, max_depth, learning_rate, num, test_size)

                print(f'Average accuracy over 5 random_states for {file} is: {avg_accuracy}')

                All_info_to_be_written.append([current_vec_title, R_n, avg_accuracy])

            el = el+1
        
        v = v+1
    
    r = r+1

All_info_to_be_written = np.array(All_info_to_be_written)

output_csv_file = f"All_Results_XGBoost.csv"
with open(output_csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Vectorization_Method_and_lead", "R_n", "Accuracy"])

    i = 0
    while i < len(All_info_to_be_written):
        row = All_info_to_be_written[i]
        writer.writerow(row)
        i = i+1

print(f'################## Saved Results to {output_csv_file} ##################')
