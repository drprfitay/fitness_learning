
import os
import pandas as pd
import numpy as np
import re

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

llm_data = {"flat": {"df": pd.read_csv("./flat_vs_mean_data/flat.csv"),
                     "emb": torch.load("./flat_vs_mean_data/flat_emb.pt")},
             "mean": {"df": pd.read_csv("./flat_vs_mean_data/mean.csv"),
                     "emb": torch.load("./flat_vs_mean_data/mean_emb.pt")}}
llm_names = ["flat", "mean"]

for i in range(1,4):
    results_dict = {}
    for llm_name in llm_names:
        sorted_emb = llm_data[llm_name]["emb"]
        sorted_llm_df = llm_data[llm_name]["df"]
        sort_idx = np.argsort(sorted_llm_df["indices"])
        sorted_llm_df = sorted_llm_df.iloc[sort_idx]
        sorted_emb = sorted_emb[sort_idx]
        llm_subset = (sorted_llm_df["mutations"] <= i).to_numpy()
        X_llm_emb_train = sorted_emb[llm_subset]
        X_llm_emb_test = sorted_emb[~llm_subset]
        y_llm_emb_train = sorted_llm_df["ground_truth"][llm_subset].to_numpy()
        y_llm_emb_test = sorted_llm_df["ground_truth"][~llm_subset].to_numpy()
        nmuts_llm_emb_test = sorted_llm_df["mutations"][~llm_subset].to_numpy()
        idx_llm = sorted_llm_df["indices"][~llm_subset].to_numpy()
        # Fit MLP for this LLM
        mlp_llm = MLPClassifier(random_state=4321, max_iter=1000)
        mlp_llm.fit(X_llm_emb_train, y_llm_emb_train)
        y_llm_pred_score = mlp_llm.predict_proba(X_llm_emb_test)[:, 1]
        y_llm_pred_label = (y_llm_pred_score > 0.5).astype(int)
    # Add to result dict with LLM-specific column names
        results_dict[f"nmuts_llm_{llm_name}"] = nmuts_llm_emb_test
        results_dict[f"idx_llm_{llm_name}"] = idx_llm
        results_dict[f"y_true_llm_{llm_name}"] = y_llm_emb_test
        results_dict[f"y_pred_llm_{llm_name}_mlp_score"] = y_llm_pred_score
        results_dict[f"y_pred_llm_{llm_name}_mlp_label"] = y_llm_pred_label
    pd.DataFrame(results_dict).to_csv(f"./flat_vs_mean_data/train_{i}.csv", index=False)

