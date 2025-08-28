#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:15:25 2025

@author: itayta
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

from constants import *
from utils import *

fix_esm_path()

import esm

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from rosetta_former.energy_vqvae import *
from dataset import *

base_dataset_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/datasets/random_100k_train"

dataset = EsmGfpDataset(sequences_path="%s/sequences.csv" % base_dataset_path,
                        embeddings_path="%s/embeddings" % base_dataset_path,
                        embedding_layer = -3,
                        tokens_path="%s/tokens" % base_dataset_path,
                        mode="embeddings")

example_x, example_y = dataset[0]

# Simulate dataset (15 samples, 1500 features each)
torch.manual_seed(42)
X_train = torch.randn(15, 1500)  # Input data
y_train = torch.randint(0, 2, (15,))  # Binary classification (adjust as needed)

# Define a simple MLP
class OverfitMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OverfitMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Initialize weights properly
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define model
input_size = example_x.shape[0]
hidden_size = 512  # Large enough to memorize
output_size = 1  # Assuming binary classification

model = OverfitMLP(input_size, hidden_size, output_size)

# Loss & optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary classification
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for item in train_data_loader:
    pass


X_train, y_train = item
y_train = y_train.argmax(dim=1)
# Training loop
epochs = 5000  # Train for a long time to overfit
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train.float())  # Ensure labels are float
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

# Check final loss (should be ~0)
print("Final Training Loss:", loss.item())



