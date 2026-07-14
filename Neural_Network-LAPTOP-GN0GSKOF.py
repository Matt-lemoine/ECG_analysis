# This file is for basic Neural Network using pytorch.

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold

def do_nn(Vec_title, N, X, y):

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64,1)
            )

        def forward(self, x):
            return self.linear_relu_stack(x)

    model = NeuralNetwork().to(device)
    print(model)
   

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    accuracies = []

    for train_index, test_index in kf.split(X, y):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test  = torch.tensor(X_test.values, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset  = TensorDataset(X_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_dataloader = DataLoader(test_dataset, batch_size=64)

        input_size = X.shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")

        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            model.train()
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


        def test(dataloader, model, loss_fn):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    pred = torch.sigmoid(pred)

                    predicted = (pred > 0.5).float()

                    correct += (predicted == y).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
            return correct


        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            accuracy = test(test_dataloader, model, loss_fn)
        print("Done!")

        accuracies.append(accuracy)

    return np.mean(accuracies)


Rns = ['R2_w7', 'R3_w7', 'R4_w7'] # If you have a specific R_n that you are interested in, specify that here.

Vectorizations = ['Betti_Vectorization', 'Pers_Vec_Vectorization', 'Pers_Vec_Vectorization_nooverlap', 'Pers_Vec_Vectorization_nocount', 'Pers_Vec_Vectorization_nocount_nooverlap', 'Pers_Vec_Vectorization_noarea']

ending_leads = ['_all', '_V1', '_V2', '_V3', '_V1_V2', '_V1_V3', '_V2_V3']

Ns = [1, 5, 10, 100]

# NN parameters
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

                    avg_accuracy = do_nn(Vec_title, n, X, y)
                    
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
                dataset = dataset[col_names]
                X = dataset.iloc[:, 0:4]
                y = dataset.iloc[:, 4].values

                avg_accuracy = do_nn(Vec_title, n, X, y)

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

                avg_accuracy = do_nn(Vec_title, n, X, y)

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

                avg_accuracy = do_nn(Vec_title, n, X, y)

                print(f'Average accuracy with 5-fold cross validation for {file} is: {avg_accuracy}')

                All_info_to_be_written.append([current_vec_title, R_n, avg_accuracy])

            el = el+1
        
        v = v+1
    
    r = r+1

All_info_to_be_written = np.array(All_info_to_be_written)

output_csv_file = f"NN_CSV_Results/All_Results_NN.csv"
with open(output_csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Vectorization_Method_and_lead", "R_n", "Accuracy"])

    i = 0
    while i < len(All_info_to_be_written):
        row = All_info_to_be_written[i]
        writer.writerow(row)
        i = i+1

print(f'################## Saved Results to {output_csv_file} ##################')
