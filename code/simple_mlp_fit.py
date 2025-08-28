#!/usr/bin/env python3
# """
# Simple MLP fitting script for embeddings to ground truth prediction
# Trains two MLPs: one on zero-shot embeddings, one on one-shot embeddings.
# Evaluates each on both zero-shot and one-shot test sets.
# """

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from math import ceil


import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

def train_trunk_mlp(data, iterations=20000, batch_size=64, lr=1e-4, save_path=None, device=torch.device("cpu")):
# Turn tensors into a dataset and dataloader
    print("\n[DEBUG] Training Trunk MLP parameters:")
    print(f"\tbase_path: {base_path}")
    print(f"\titerations: {iterations}")
    print(f"\tbatch_size: {batch_size}")
    print(f"\tlr: {lr}")
    print(f"\tsave_path: {save_path}")
    print(f"\tdevice: {device}")

    X_train = data["X_train"] 
    y_train = data["y_train"] 
    X_test = data["X_test"] 
    y_test = data["y_test"]
    
    #load_embeddings_and_labels(base_path)
    print(f"\tTrain set size: {X_train.shape[0]}")
    print(f"\tTest set size: {X_test.shape[0]}\n")


    y_train = torch.nn.functional.one_hot(y_train.to(torch.long), 2).to(torch.float)
    y_test = torch.nn.functional.one_hot(y_test.to(torch.long), 2).to(torch.float)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=256)

    # Define a configurable ReLU MLP
    class TrunkMLP(nn.Module):
        def __init__(self, input_dim, hidden_layer_sizes, output_dim):
            super().__init__()
            layers = []
            prev_dim = input_dim

            for h in hidden_layer_sizes:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                prev_dim = h

            layers.append(nn.Linear(prev_dim, output_dim))

            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    # Configurable hidden layer sizes
    hidden_layer_sizes = [512, 256]  # Example: two hidden layers with 64 and 32 units
    n_epochs = ceil(iterations / len(train_dataloader))

    model = TrunkMLP(input_dim=X_train.shape[1], hidden_layer_sizes=hidden_layer_sizes, output_dim=2).to(device)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    

    model.train()

    running_batch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_epoch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_20b_loss = torch.tensor([], dtype=torch.float).to(device)
    # Instantiate model, loss, optimizer

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = torch.tensor(0.0).to(device)
        iter_20b_loss = torch.tensor(0.0).to(device)
        for step, batch, in enumerate(train_dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce_loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iter_20b_loss += loss.item()
            running_batch_loss = torch.cat([running_batch_loss, loss.detach().reshape(-1)])
            
            if (step + 1) % 20 == 0:
                iter_20b_loss = iter_20b_loss / 20
                running_20b_loss = torch.cat([running_20b_loss, iter_20b_loss.detach().reshape(-1)])
                iter_20b_loss = torch.tensor(0, dtype=torch.float).to(device)
                plt.plot(range(1, running_20b_loss.shape[0] + 1), running_20b_loss.cpu().detach().numpy())
                plt.draw()
                plt.pause(0.001)
                plt.close()
                
            print("[E%d I%d] %.3f" % (epoch, step, loss))
        running_epoch_loss = torch.cat([running_epoch_loss, epoch_loss.detach().reshape(-1)])


        # INSERT_YOUR_CODE
        # After training, evaluate on train and test sets

        # model.eval()
        # with torch.no_grad():
        #     # Evaluate on train set
        #     train_logits = model(torch.tensor(X_train, dtype=torch.float32).to(device))
        #     train_pred = train_logits.argmax(dim=1).cpu().numpy()
        #     train_acc = np.mean(train_pred == y_train)
        #     print(f"Train accuracy: {train_acc:.4f}")

        #     # Evaluate on test set
        #     test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        #     test_pred = test_logits.argmax(dim=1).cpu().numpy()
        #     test_acc = np.mean(test_pred == y_test)
        #     print(f"Test accuracy: {test_acc:.4f}")

        #     # Optionally, print confusion matrix
        #     print("Train confusion matrix:")
        #     print(confusion_matrix(y_train, train_pred))
        #     print("Test confusion matrix:")
        #     print(confusion_matrix(y_test, test_pred))

    # eval loop
    predicted_score = torch.tensor([], dtype=torch.float) 
    predicted_label = torch.tensor([], dtype=torch.float)
    with torch.no_grad():
        for step, batch, in enumerate(test_dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = logits.softmax(dim=1)[:,0]
            label = logits.argmax(dim=1).float()
            predicted_score = torch.cat([predicted_score, probs.cpu().detach().reshape(-1)])
            predicted_label = torch.cat([predicted_label, label.cpu().detach().reshape(-1)])


    gt_label = y_test.argmax(dim=1)
    sorted_seq = np.argsort(-predicted_score)
    top_K_pct = dict([("%d" % K, np.unique(gt_label[sorted_seq[0:K]], return_counts=True)[1][0]/K) for K in [5,10,50,100,500,1000,5000]])

    # Create a DataFrame with predicted_score and predicted_label
    df_pred = pd.DataFrame({
        "predicted_score": predicted_score.numpy(),
        "predicted_label": predicted_label.numpy(),
        "ground_truth_label": y_test.argmax(dim=1).numpy(),
    })


    
    loss_dict = {
        "running_batch_loss": running_batch_loss.cpu().numpy(),
        "running_epoch_loss": running_epoch_loss.cpu().numpy(),
        "running_20b_loss": running_20b_loss.cpu().numpy()
    }

    for loss_name, loss_array in loss_dict.items():
        loss_path = os.path.join(base_path, f"trunk_mlp_{loss_name}.npy")
        np.save(loss_path, loss_array)
        print(f"Saved {loss_name} to {loss_path}")

    csv_path = os.path.join(base_path, "trunk_mlp_predictions.csv")
    df_pred.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

# path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/pretraining/triplet_loss_backbones/zero_shot/train_12_test_4"
# train_trunk_mlp(path)

