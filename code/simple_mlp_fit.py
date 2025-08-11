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

# from torch.types import Device

# # Define shared base path for pretraining data
# pretraining_base_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/pretraining/triplet_loss_backbones"

# # Load zero-shot train data
# zs_embeddings = torch.load(f"{pretraining_base_path}/zero_shot/train/embeddings.pt")
# zs_ground_truth = torch.load(f"{pretraining_base_path}/zero_shot/train/ground_truth.pt")

# # Load one-shot train data
# os_embeddings = torch.load(f"{pretraining_base_path}/one_shot/train/embeddings.pt")
# os_ground_truth = torch.load(f"{pretraining_base_path}/one_shot/train/ground_truth.pt")

# base_results_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/results/from_wexac_eval/gfp/train_1_2_3"
# base_results_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/pretraining/triplet_loss_backbones/zero_shot/train_1_test_3"

# # One-shot folders
# oneshot_test_folder = "test_4/one_shot/test"
# oneshot_test_folder = "random_50k/one_shot"

# # Zero-shot folders
# zeroshot_test_folder = "test_4/zero_shot/test"
# zeroshot_test_folder = "random_50k/zero_shot"

# # One-shot test embeddings and ground truth
# test_embeddings_oneshot = torch.load(f"{base_results_path}/{oneshot_test_folder}/embeddings.pt")
# test_ground_truth_oneshot = torch.load(f"{base_results_path}/{oneshot_test_folder}/ground_truth.pt")

# # Zero-shot test embeddings and ground truth
# test_embeddings_zeroshot = torch.load(f"{base_results_path}/{zeroshot_test_folder}/embeddings.pt")
# test_ground_truth_zeroshot = torch.load(f"{base_results_path}/{zeroshot_test_folder}/ground_truth.pt")



# def to_numpy(x):
#     if isinstance(x, torch.Tensor):
#         return x.cpu().numpy()
#     return np.array(x)

# # Convert all to numpy arrays
# zs_embeddings = to_numpy(zs_embeddings)
# zs_ground_truth = to_numpy(zs_ground_truth)
# os_embeddings = to_numpy(os_embeddings)
# os_ground_truth = to_numpy(os_ground_truth)
# test_embeddings_oneshot = to_numpy(test_embeddings_oneshot)
# test_ground_truth_oneshot = to_numpy(test_ground_truth_oneshot)
# test_embeddings_zeroshot = to_numpy(test_embeddings_zeroshot)
# test_ground_truth_zeroshot = to_numpy(test_ground_truth_zeroshot)

# def evaluate_mlp(mlp, X_train, y_train, X_test, y_test, test_name="", K=500):
#     # Fit
#     mlp.fit(X_train, y_train)
#     # Predict
#     train_pred = mlp.predict(X_train)
#     test_pred = mlp.predict(X_test)
#     # Confusion matrix
#     cm = confusion_matrix(y_test, test_pred)
#     # Calculate FPR/FNR
#     if cm.shape == (2,2):
#         tn, fp, fn, tp = cm.ravel()
#         false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
#         false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
#     else:
#         false_positive_rate = false_negative_rate = float('nan')
#     # Accuracy
#     train_acc = mlp.score(X_train, y_train)
#     test_acc = mlp.score(X_test, y_test)

#     #top_sequences = np.unique(y_test[np.argsort(-mlp.predict_proba(X_test)[:,0])][0:K], return_counts=True)
#     sorted_seq = np.argsort(-mlp.predict_proba(X_test)[:,0])

#     top_K_pct = dict([("%d" % K, np.unique(y_test[sorted_seq[0:K]], return_counts=True)[1][0]/K) for K in [5,10,50,100,500,1000,5000]])

#     # Print
#     print(f"\n=== Evaluation: {test_name} ===")
#     print(f"Train accuracy: {train_acc:.4f}")
#     print(f"Test accuracy: {test_acc:.4f}")
#     print(f"False Positive Rate: {false_positive_rate:.4f}")
#     print(f"False Negative Rate: {false_negative_rate:.4f}")
#     print("Confusion Matrix:")
#     print(cm)

#     return {
#         "train_acc": train_acc,
#         "test_acc": test_acc,
#         "fpr": false_positive_rate,
#         "fnr": false_negative_rate,
#         "cm": cm,
#         "top_K": top_K_pct
#     }

# # Define MLPs
# mlp_zs = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
# mlp_os = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# # 1. Train on zero-shot, evaluate on both test sets
# evaluate_mlp(
#     mlp_zs,
#     zs_embeddings, zs_ground_truth,
#     test_embeddings_zeroshot, test_ground_truth_zeroshot,
#     test_name="Zero-shot MLP on Zero-shot Test"
# )


# # 2. Train on one-shot, evaluate on both test sets
# evaluate_mlp(
#     mlp_os,
#     os_embeddings, os_ground_truth,
#     test_embeddings_oneshot, test_ground_truth_oneshot,
#     test_name="One-shot MLP on One-shot Test"
# )



import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

def load_embeddings_and_labels(base_path):
    """
    Loads embeddings and ground truth labels from train and test folders under base_path.
    Returns: X_train, y_train, X_test, y_test (all torch tensors)
    """
    data = {}
    for split in ['train', 'test']:
        split_path = os.path.join(base_path, split)
        X = torch.load(os.path.join(split_path, "embeddings.pt"))
        y = torch.load(os.path.join(split_path, "ground_truth.pt"))
        data[f"X_{split}"] = X
        data[f"y_{split}"] = y
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

# Example usage:
# base_path = "/path/to/data"
# 


def train_trunk_mlp(base_path, iterations=20000, batch_size=64, lr=1e-4, save_path=None, device=torch.device("cpu")):
# Turn tensors into a dataset and dataloader
    X_train, y_train, X_test, y_test = load_embeddings_and_labels(base_path)

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
        "predicted_label": predicted_label.numpy()
    })

    csv_path = os.path.join(base_path, "trunk_mlp_predictions.csv")
    df_pred.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

# path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/pretraining/triplet_loss_backbones/zero_shot/train_12_test_4"
# train_trunk_mlp(path)