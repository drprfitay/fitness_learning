
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
from sklearn.metrics import confusion_matrix

from sequence_space_utils import *


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


def collect_data(path, sequence_df_path, nmuts_column, train_mutations=None, test_mutations=None):
    """
    Aggregates all train_X and test_Y subfolders into a single dataframe, 
    and collects the number of mutations for each.
    Skips subdirs with duplicate mutation numbers.
    Returns:
        df: pd.DataFrame with columns ['split', 'mutations', 'ground_truth', 'indices']
        embeddings: torch.Tensor of all embeddings (row order matches df)
        train_mutations: list of mutation numbers used for train
        test_mutations: list of mutation numbers used for test
        df_train: subset of df for train_mutations
        df_test: subset of df for test_mutations
        X_train: torch.Tensor of embeddings for train
        y_train: pd.DataFrame with ground_truth and indices for train
        X_test: torch.Tensor of embeddings for test
        y_test: pd.DataFrame with ground_truth and indices for test
    """
    all_rows = pd.DataFrame(columns=["split", "mutations", "ground_truth", "indices"])
    all_embeddings = []
    sequence_df = pd.read_csv(sequence_df_path)
    mutations_aggregated = []

    # Regex to extract train and test mutation numbers from subdir name
    # Example: train_1x2_test_3x4 or train_1x2_test_12
    pair_re = re.compile(r"^train_([0-9x]+)_test_([0-9x]+)$")

    for subdir in os.listdir(path):
        subpath = os.path.join(path, subdir)
        if not os.path.isdir(subpath):
            continue

        match = pair_re.match(subdir)
        if not match:
            continue

        train_mut_str = match.group(1)  # e.g. "1x2"
        test_mut_str = match.group(2)   # e.g. "3x4" or "12"

        # Optionally, split by 'x' to get list of mutation numbers as strings
        train_mutations = train_mut_str.split('x')
        test_mutations = test_mut_str.split('x')

        for split in ["train", "test"]:
            split_path = os.path.join(subpath, split)
            emb_path = os.path.join(split_path, "embeddings.pt")
            gt_path = os.path.join(split_path, "ground_truth.pt")
            idx_path = os.path.join(split_path, "indices.pt")

            if not (os.path.exists(emb_path) and os.path.exists(gt_path) and os.path.exists(idx_path)):
                continue

            muts = train_mut_str if split == "train" else test_mut_str

            emb = torch.load(emb_path)
            gt = torch.load(gt_path)
            idx = torch.load(idx_path)
            seq_df_subset = sequence_df.iloc[idx.numpy()]
            muts_in_data = pd.unique(seq_df_subset[nmuts_column])

            # sanity
            assert sum(np.isin(muts_in_data, np.array(muts.split('x')).astype(int))) == len(muts_in_data)

            for m in muts_in_data:
                if m in mutations_aggregated:
                    print("\tAlready aggregated mutation %d - continuing" % m )
                    continue

                indices_to_use = (seq_df_subset[nmuts_column] == m).to_numpy()
                # Sanity X2
                assert pd.unique(sequence_df.iloc[idx[indices_to_use].numpy()][nmuts_column]) == m
                mutations_aggregated.append(m)

                # emb: [N, D], gt: [N], idx: [N]
                n = sum(indices_to_use)
                # Save the original mutation strings as requested
                df_tmp = pd.DataFrame({
                    "split": [split] * n,
                    "mutations": [m] * n,
                    "ground_truth": gt[indices_to_use].numpy(),
                    "indices": idx[indices_to_use].numpy()
                })

                all_rows = pd.concat([all_rows, df_tmp], ignore_index=True)
                all_embeddings.append(emb[indices_to_use])

    if len(all_embeddings) > 0:
        all_embeddings = torch.cat(all_embeddings, dim=0)
    else:
        all_embeddings = torch.empty((0,))
    
    df = pd.DataFrame(all_rows)

    return df, all_embeddings

     

base_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
base_results_path = "%s/results/gfp_dataset/llm_epinnet_comparisions" % base_path
sequence_df_path = "%s/data/configuration/fixed_unique_gfp_sequence_dataset_full_seq.csv" % base_path
llm_evaluation_paths = {
    "esm2_8m": "%s/results/from_wexac_eval/gfp/esm8m/zero_shot" % base_path,
    "esm2_35m": "%s/results/from_wexac_eval/gfp/esm35m/zero_shot" % base_path,
    "esm2_650m": "%s/results/from_wexac_eval/gfp/esm650m/zero_shot" % base_path
}
nmuts_column = "num_muts"

mutations_to_use = range(1, 12)
seq_df = pd.read_csv(sequence_df_path)
seq_df = seq_df[np.isin(seq_df[nmuts_column].to_numpy(), np.array(mutations_to_use))]  # subset by mutations
one_hot = get_one_hot_encoding(seq_df, "L42", "V224")

# Collect and sort LLM data for each model
llm_data = {}
for llm_name, llm_path in llm_evaluation_paths.items():
    df, emb = collect_data(llm_path, sequence_df_path, nmuts_column)
    sort_idx = np.argsort(df["indices"])
    llm_data[llm_name] = {
        "df": df.iloc[sort_idx].reset_index(drop=True),
        "emb": emb[sort_idx]
    }

# Make sure all LLMs are aligned with seq_df
for llm_name in llm_data:
    sorted_llm_df = llm_data[llm_name]["df"]
    assert np.all(seq_df[nmuts_column].to_numpy() == sorted_llm_df["mutations"].to_numpy())
    assert np.all(seq_df["inactive"].astype(int).to_numpy() == sorted_llm_df["ground_truth"].to_numpy())

all_results = []

# Train on I test on the rest
for i in range(1, 6):

    ohe_subset = (seq_df[nmuts_column] <= i).to_numpy()

    # Prepare all relevant arrays for one-hot
    X_ohe_train = one_hot[ohe_subset]
    X_ohe_test = one_hot[~ohe_subset]
    y_ohe_train = seq_df["inactive"].astype(int).to_numpy()[ohe_subset]
    y_ohe_test = seq_df["inactive"].astype(int).to_numpy()[~ohe_subset]
    nmuts_ohe_train = seq_df[nmuts_column][ohe_subset].to_numpy()
    nmuts_ohe_test = seq_df[nmuts_column][~ohe_subset].to_numpy()

    # Fit one-hot models
    mlp_ohe = MLPClassifier(random_state=4321, max_iter=1000)
    mlp_ohe.fit(X_ohe_train, y_ohe_train)
    y_ohe_pred_score = mlp_ohe.predict_proba(X_ohe_test)[:, 1]
    y_ohe_pred_label = (y_ohe_pred_score > 0.5).astype(int)

    linreg_ohe = LinearRegression()
    linreg_ohe.fit(X_ohe_train, y_ohe_train)
    y_ohe_linreg_pred_score = linreg_ohe.predict(X_ohe_test)
    y_ohe_linreg_pred_label = (y_ohe_linreg_pred_score > 0.5).astype(int)

    # Prepare result dict
    result_dict = {
        "train_max_mut": [i] * len(nmuts_ohe_test),
        "nmuts_seq": nmuts_ohe_test,
        "idx_seq": seq_df.index[~ohe_subset].to_numpy(),
        "y_true_seq": y_ohe_test,
        "y_pred_ohe_mlp_score": y_ohe_pred_score,
        "y_pred_ohe_mlp_label": y_ohe_pred_label,
        "y_pred_ohe_linreg": y_ohe_linreg_pred_score,
        "y_pred_ohe_linreg_label": y_ohe_linreg_pred_label
    }

    # For each LLM, fit and predict, and add to result dict
    for llm_name in ["esm2_8m", "esm2_35m", "esm2_650m"]:
        sorted_emb = llm_data[llm_name]["emb"]
        sorted_llm_df = llm_data[llm_name]["df"]
        llm_subset = (sorted_llm_df["mutations"] <= i).to_numpy()
        assert np.all(llm_subset == ohe_subset)

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
        result_dict[f"nmuts_llm_{llm_name}"] = nmuts_llm_emb_test
        result_dict[f"idx_llm_{llm_name}"] = idx_llm
        result_dict[f"y_true_llm_{llm_name}"] = y_llm_emb_test
        result_dict[f"y_pred_llm_{llm_name}_mlp_score"] = y_llm_pred_score
        result_dict[f"y_pred_llm_{llm_name}_mlp_label"] = y_llm_pred_label

    # Build dataframe for this split
    results_df = pd.DataFrame(result_dict)
    # Save each df separately as train_{i}.csv
    results_path = f"{base_results_path}/train_{i}.csv"
    results_df.to_csv(results_path, index=False)
    

