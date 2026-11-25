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
from sklearn.metrics import roc_auc_score

def do_nn(X, y, input_size, learning_rate, batch_size, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 8),
                nn.ReLU(),
                nn.Linear(8,1)
            )

        def forward(self, x):
            return self.linear_relu_stack(x)

    model = NeuralNetwork().to(device)
    print(model)
   

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    auc_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start = 1):

        print(f"\n Starting fold {fold} of 5")

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

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            model.train()
            for batch, (X, y) in enumerate(dataloader):
                print(batch)

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
            num_batches = len(dataloader)
            model.eval()
            test_loss = 0

            y_true = []
            y_score = []

            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)

                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()

                    pred = torch.sigmoid(pred)

                    y_true.extend(y.cpu().numpy().ravel())
                    y_score.extend(pred.cpu().numpy().ravel())

            test_loss /= num_batches

            auc = roc_auc_score(y_true, y_score)

            print(f"Test Error: \n ROC-AUC: {auc:.4f}, Avg loss: {test_loss:>8f} \n")
        
            return auc


        epochs = epochs
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            auc = test(test_dataloader, model, loss_fn)

        print("Done!")

        auc_scores.append(auc)



    # Now for the same thing but with Accuracy instead of AUC-ROC

    accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start = 1):

        print(f"\n Starting fold {fold} of 5")

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

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            model.train()
            for batch, (X, y) in enumerate(dataloader):
                print(batch)

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

        epochs = epochs
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            accuracy = test(test_dataloader, model, loss_fn)
        print("Done!")

        accuracies.append(accuracy)

    return np.array(auc_scores), np.array(accuracies)



Rns = ['R2', 'R3', 'R4'] # If you have a specific R_n that you are interested in, specify that here.

Vectorizations = ['Betti_Vectorization', 'Pers_Vec_Vectorization', 'Pers_Vec_Vectorization_nooverlap', 'Pers_Vec_Vectorization_nocount', 'Pers_Vec_Vectorization_nocount_nooverlap', 'Pers_Vec_Vectorization_noarea']

ending_leads = ['_all', '_V1', '_V2', '_V3', '_V1_V2', '_V1_V3', '_V2_V3']

Ns = [1, 5, 10, 100]

# NN parameters
learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [32, 64, 128]
epochs_list = [5, 10, 20]
# learning_rate = 1e-3, batch_size = 64, epochs = 5 ORIGNIAL


l = 0
while l < len(learning_rates):
    learning_rate = learning_rates[l]

    b = 0
    while b < len(batch_sizes):
        batch_size = batch_sizes[b]

        e = 0
        while e < len(epochs_list):

            epochs = epochs_list[e]


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
                                N = Ns[i]
    
                                csv_name = Vec_title + suffix_lead + '_' + str(N) + '_' + str(R_n) + '.csv'
    
                                file = f'Vectorizations_10_16_26/{csv_name}'
    
                                print(file)
    
                                dataset = pd.read_csv(file, header = None)
                                print(dataset)
                                dataset = dataset.drop(dataset.columns[[0]], axis = 1)
                                max_col = dataset.shape[1]-1
                                X = dataset.iloc[:, 0:max_col]
                                y = dataset.iloc[:, max_col].values

                                input_size = np.shape(X)[1]
                                print(input_size)
    
                                auc_scores, accuracy_scores = do_nn(X, y, input_size, learning_rate, batch_size, epochs)
                                mean_auc_score = auc_scores.mean()
                                std_auc_score = auc_scores.std()
                                mean_accuracy_score = accuracy_scores.mean()
                                std_accuracy_score = accuracy_scores.std()
                                
                                print(f'Average AUC over 5 fold cross validation for {file} with N = {N} is: {mean_auc_score}')
                                print(f'Average AUC over 5 fold cross validation for {file} with N = {N} is: {mean_accuracy_score}')
    
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
    
                            input_size = np.shape(X)[1]
                            print(input_size)
    
                            auc_scores, accuracy_scores = do_nn(X, y, input_size, learning_rate, batch_size, epochs)
                            mean_auc_score = auc_scores.mean()
                            std_auc_score = auc_scores.std()
                            mean_accuracy_score = accuracy_scores.mean()
                            std_accuracy_score = accuracy_scores.std()
    
                            print(f'Average AUC over 5 fold cross validation for {file} is: {mean_auc_score}')
                            print(f'Average AUC over 5 fold cross validation for {file} is: {mean_accuracy_score}')
    
                            All_info_to_be_written.append([Vec_title + suffix_lead, R_n, mean_auc_score, std_auc_score, mean_accuracy_score, std_accuracy_score])
                            
                        el = el+1
                    
                    v = v+1
                
                r = r+1

            All_info_to_be_written = np.array(All_info_to_be_written)

            output_csv_file = f"All_Results_NN_lr_{learning_rate}_bs_{batch_size}_epochs_{epochs}.csv"
            with open(output_csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Vectorization_Method_and_lead", "R_n", "Mean_AUC", "STD_AUC", "Mean_Accuracy", "STD_Accuracy"])
                
                i = 0
                while i < len(All_info_to_be_written):
                    row = All_info_to_be_written[i]
                    writer.writerow(row)
                    i = i+1

            print(f'################## Saved Results to {output_csv_file} ##################')
        
            e = e+1

        b = b+1

    l = l+1
