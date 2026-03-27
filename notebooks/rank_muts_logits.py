import pandas as pd
import numpy as np
import os
import torch
import seaborn as sns
import random

from scipy.stats import pearsonr, spearmanr
from scipy.stats import rankdata
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from Bio import SeqIO

from random import sample

from utils_for_analysis import (
    calculate_ss_for_df_and_factors,
    load_df_all,
    load_df_with_budget,
    discretized_parameter_scale,
    xlabel_dict,
    ylabel_dict,
    title_fontsize,
    label_fontsize,
    tick_fontsize,
    legend_fontsize,
    original_parameter_scale,
    color_map,
    fix_ticks,
    get_labels,
    positions,
    num_muts_column_name,
    DATASET_PATHS
)


def get_relevant_columns_gfp_protgym(df, first_col, last_col):
    si = np.where(df.columns == first_col)[0][0]
    ei = np.where(df.columns == last_col)[0][0]+1
    return df.columns[si:ei]

def nmt_relevant_columns(df):
    si = np.where(df.columns == "1")[0][0]
    ei = np.where(df.columns == "272")[0][0]+1
    positions_with_mutations =  np.array([len(pd.unique(df.iloc[:,i])) > 1 for i in range(si,ei)])
    return df.columns[si:ei][positions_with_mutations]

def get_relevant_columns(dataset, df):
    cols = get_relevant_columns_gfp_protgym(df, positions[dataset][0], positions[dataset][1])
    return cols

def get_num_muts_column_name(dataset):
    return {
        "pard3": "num_muts",
        "lov": "num_muts",
        "gfp": "num_muts",
        "pte": "num_muts",
        "gcn4": "num_muts",
        "nmt": "num_muts",
        "aamyl": "num_muts"
    }[dataset]

# --- OHE Feature Creation ---
def get_one_hot_encoding(sdf, relevant_columns):
    one_hot_encoding = pd.get_dummies(sdf[relevant_columns])
    return one_hot_encoding

datasets_and_activity = {
    "gcn4": {"path": "./data/gcn4/gcn4.csv", "activity_col": "activity"},
    "pard3": {"path": "./data/pard3/pard3.csv", "activity_col": "activity"},
    "pte": {"path": "./data/pte/pte.csv", "activity_col": "p-nitrophenyl_octanoate"},
    "nmt": {"path": "./data/nmt/nmt_full_seq.csv", "activity_col": "activity"},
    "lov": {"path": "./data/lov/lov.csv", "activity_col": "activity"},
    "gfp": {"path": "./data/gfp/gfp_dataset_10mut.csv", "activity_col": "inactive"},
    "aamyl": {"path": "./data/aamyl/aamyl.csv", "activity_col": "activity"}
}

import sys
os.chdir(os.path.join(os.getcwd(), "../code/"))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from plm_base import *
plm_init(os.path.join(os.getcwd(), "../"))

os.chdir(os.path.join(os.getcwd(), "../notebooks"))

model_to_use = "esm2_t33_650M_UR50D"
model = plmEmbeddingModel(plm_name="%s" % model_to_use, logits_only=True, emb_only=False).eval()

def get_one_hot_encoding(sdf, relevant_columns):
    one_hot_encoding = pd.get_dummies(sdf[relevant_columns])
    return one_hot_encoding

for dataset in ["aamyl"]:
    dataset_path = DATASET_PATHS[dataset]
    df = pd.read_csv(dataset_path)
    wt_seq = df[df["num_muts"] == 0]["full_seq"].iloc[0]

    # get one-hot encoding for relevant columns
    relevant_columns = get_relevant_columns(dataset, df)
    ohe = get_one_hot_encoding(df, relevant_columns)
    ohe_columns = ohe.columns
    ohe = ohe.to_numpy().astype(int)
    assert ohe.shape[1] == sum([len(pd.unique(df[C])) for C in relevant_columns])

    activity_col = datasets_and_activity[dataset]["activity_col"]
    activity = df[activity_col].to_numpy()
    if dataset == "gfp":
        activity = (~activity).astype(int)

    mean_with_mutation_per_column = []
    mean_without_mutation_per_column = []
    for i in range(ohe.shape[1]):
        mean_val_with_mutation = np.mean(activity[np.where(ohe[:, i] == 1)[0]])
        mean_val_without_mutation = np.mean(activity[np.where(ohe[:, i] == 0)[0]])
        mean_with_mutation_per_column.append(mean_val_with_mutation/mean_val_without_mutation)
        mean_without_mutation_per_column.append(mean_val_without_mutation/mean_val_with_mutation)



    plt.scatter(mean_without_mutation_per_column, mean_with_mutation_per_column)
    plt.xlabel("Mean without mutation")
    plt.ylabel("Mean with mutation")
    plt.show()

    plt.hist(np.array(mean_with_mutation_per_column)/np.array(mean_without_mutation_per_column), bins=20)


    pssm = pd.read_csv(f"./data/{dataset}//pssm_scores.csv")
    ratio = np.array(mean_with_mutation_per_column)/np.array(mean_without_mutation_per_column)

    K1_dict = {
        "gcn4": 160,
        "lov": 6,
        "gfp": 7,
        "nmt": 15,
        "pte": 2,
        "pard3": 6,
        "aamyl": 6
    }

    K2_dict = {
        "gcn4": {"best": 5, "worst": 2},
        "lov": {"best": 5, "worst": 4},
        "gfp": {"best": 3, "worst": 6},
        "nmt": {"best": 12, "worst": 1},
        "pte": {"best": 1, "worst": 1},
        "pard3": {"best": 2, "worst": 1},
        "aamyl": {"best": 2, "worst": 1}
    }

    worst_four = np.argsort(ratio)[0:K1_dict[dataset]]
    best_four = np.argsort(-ratio)[0:K1_dict[dataset]]

    from_best = np.array(ohe[:, best_four].sum(axis=1) >= K2_dict[dataset]["best"])
    from_worst = np.array(ohe[:, worst_four].sum(axis=1) >= K2_dict[dataset]["worst"])
    print(sum(from_best))
    print(sum(from_worst))


    from_best_activities = []
    from_worst_activities = []
    for _ in range(20):
        best_indices = np.where(from_best)[0]
        worst_indices = np.where(from_worst)[0]

        if len(best_indices) >= 30:
            sampled_best = np.random.choice(best_indices, 30, replace=False)
        else:
            sampled_best = np.random.choice(best_indices, 30, replace=True)
        if len(worst_indices) >= 30:
            sampled_worst = np.random.choice(worst_indices, 30, replace=False)
        else:
            sampled_worst = np.random.choice(worst_indices, 30, replace=True)

        mean_best = np.mean(activity[sampled_best])
        mean_worst = np.mean(activity[sampled_worst])

        from_best_activities.append(mean_best)
        from_worst_activities.append(mean_worst)

    print("Mean activity for from_best, 20 runs: %.3f", np.mean(from_best_activities))
    print("Mean activity for from_worst, 20 runs: %.3f", np.mean(from_worst_activities))

    best_four = ohe_columns[best_four]
    worst_four = ohe_columns[worst_four]

    pos_per_best = [int(a.split("_")[0][1:]) for a in best_four.to_list()]
    pos_per_worst = [int(a.split("_")[0][1:]) for a in worst_four.to_list()]
    vocab_per_best = [a.split("_")[-1] for a in best_four.to_list()]
    vocab_per_worst = [a.split("_")[-1] for a in worst_four.to_list()]

    pssm_scores_best = []
    pssm_scores_worst = []
    for p,v in zip(pos_per_best, vocab_per_best):
        pssm_scores_best.append(pssm[pssm["position"] == p][v].iloc[0])

    for p,v in zip(pos_per_worst, vocab_per_worst):
        pssm_scores_worst.append(pssm[pssm["position"] == p][v].iloc[0])


    print(pos_per_worst)
    print(pos_per_best)

    logits_path = f"./data/{dataset}/%s_logits.np.npy" % model_to_use
    
    overwrite = True
    if os.path.exists(logits_path) and not overwrite:
        print("Loading logits from %s" % logits_path)
        logits = np.load(logits_path)
        print("Loaded logits from %s" % logits_path)
    else:
        vocab = model.vocab
        all_tokens = model.encode("".join(vocab))
        wt_tokens = model.encode(wt_seq)
        all_tokens = all_tokens[1:-1]
        model.to("cuda")
        logits = model(torch.tensor(wt_tokens, device="cuda").unsqueeze(0))
        logits = logits[0].softmax(dim=1)
        logits = logits.cpu().detach().numpy()
        np.save(logits_path, logits)
        print("Saved logits to %s" % logits_path)

    model_logits = logits
    