#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:28:01 2025

@author: itayta
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import string


# Plotting variance explained bars (following generate_budgeted_subsamples style)
import numpy as np
import pandas as pd
import os
import glob
import torch


from pathlib import Path

import matplotlib.pyplot as plt


from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc
)


root_dataset_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/data/merged_datasets"

DATASET_PATHS = {
    "gfp" : "data/gfp/gfp_dataset_10mut.csv",
    "lov": "data/lov/lov.csv",
    "pard3": "data/pard3/pard3.csv",
    "gcn4": "data/gcn4/gcn4.csv",
    "pte": "data/pte/pte.csv",
    "nmt": "data/nmt/nmt.csv",
    "aamyl": "data/aamyl/aamyl.csv",
    "trpb": "data/trpb/trpb.csv",
    "ired": "data/ired/ired.csv",
    "nucb": "data/nucb/nucb.csv",
    "his": "data/his/his.csv",
    "his2": "data/his2/his2.csv",
    "his5": "data/his5/his5.csv",
    "casp": "data/casp/casp.csv"
}

positions = {
    "nmt": ["Y20", "F253"],
    "gfp": ["L42", "V224"],
    "lov": ["G2", "T112"],
    "pard3": ["L48","R82"],
    "gcn4": ["S101","S144"],
    "pte": ["I72", "M283"],
    "aamyl": ["P4", "D425"],
    "trpb": ["A104", "Y301"],
    "ired": ["M1", "S290"],
    "nucb": ["Y25", "Q142"],
    "his": ["L7", "D211"],
    "his2": ["N137", "F163"],
    "his5": ["R181", "D211"],
    "casp": ["D561", "R588"]
}

xlabel_dict = {
    'model_name': 'Model',
    'scale': 'Model parameters',
    'train_mutations': 'Mutations in train'
}
ylabel_dict = {
    "roc": "ROC AUC",
    "precision": "Precision",
    "f1": "F1 Score",
    "accuracy": "Accuracy",
    "recall": "Recall",
    "top_100_pct": "Precision@100",
    "correlation": "Spearman's ρ",
    "cor_llm": "Spearman's ρ"
}

color_map = {
    "esm_8m": "#002b87",
    "esm_35m": "#f0b800",
    "esm_150m": "#008230",    
   "esm_650m": "#bd0000",
    "esm_3b": "#00cfeb",
    "progen2-small": "#ffa500",
    "progen2-medium": "#a603a0",
    "prot_bert": "#3d546b"
}

original_parameter_scale = {"esm_35m": 35,
                            "esm_8m": 8,
                            "esm_150m": 150,
                            "esm_650m": 650,
                            "progen2-small": 151,
                            "progen2-medium": 764,
                            "prot_bert": 420}

full_seq_column_name = {
    "gcn4": "full_seq",
    "pard3": "full_seq",
    "lov": "full_seq",
    "gfp": "full_seq",
    "pte": "full_seq",   
    "aamyl": "full_seq",
    "nmt": "seq",
    "trpb": "full_seq",
    "ired": "full_seq",
    "nucb": "full_seq",
    "his": "full_seq",
    "his2": "full_seq",
    "his5": "full_seq",
    "casp": "full_seq"
}

num_muts_column_name = {
    "pard3": "num_muts",
    "lov": "num_muts",
    "gfp": "num_muts",
    "pte": "num_muts",   
    "gcn4": "num_muts",
    "nmt": "num_muts",
    "aamyl": "num_muts",
    "trpb": "num_muts",
    "ired": "num_muts",
    "nucb": "num_muts",
    "his": "num_muts",
    "his2": "num_muts",
    "his5": "num_muts",
    "casp": "num_muts"
}


discretized_parameter_scale = {"esm_35m": "0_100",
                                "esm_8m": "0_100",
                                "esm_150m": "100_500",
                                "esm_650m": "500_1000",
                                "esm_3b": "500_1000",
                                "progen2-small": "100_500",
                                "progen2-medium": "500_1000",
                                "prot_bert": "100_500"}
title_fontsize = 9
label_fontsize = 8
tick_fontsize = 7
legend_fontsize = 7


def mean_without_nan(statistics_matrix):
    mu_vec = np.apply_along_axis(lambda col: np.nanmean(col), axis=0, arr=statistics_matrix)
    return mu_vec

def calculate_ss_for_df_and_factors(df, 
    factors = ["model_name", "train_mutations"], 
    variables_to_calculate = ["roc", "top_100_pct", "precision", "f1", "accuracy", "recall"]
):
    results = []
    ss_all = []

    for var in variables_to_calculate:
        SS_ALL = (df[var] - df[var].mean()) ** 2
        for factor in factors:
            groups_in_factor = df[factor].unique()
            for group in groups_in_factor:
                group_df = df[df[factor] == group]
                SS = (group_df[var] - group_df[var].mean()) ** 2
                results.append({
                    'factor_name': factor,
                    'SS': SS.sum(),
                    'group': group,
                    'var': var
                })

        ss_all.append({"var": var, "SS": SS_ALL.sum()})

    results_df = pd.DataFrame(results)
    ss_all = pd.DataFrame(ss_all)

    return ss_all, results_df

def load_df_all(embedding_base_dir, one_hot=False):

    df_all_list = []

    experiment_dirs = [d for d in os.listdir(embedding_base_dir) if os.path.isdir(os.path.join(embedding_base_dir, d))]
    llms_experiment_dirs = [e for e in experiment_dirs if e != "one_hot"]

    for exp_subdir in llms_experiment_dirs:
        exp_path = os.path.join(embedding_base_dir, exp_subdir)
        csv_files = [f for f in os.listdir(exp_path) if f.endswith('.csv')]
        print(f"\nProcessing model directory: {exp_subdir} at {exp_path}")
        print(f"Found {len(csv_files)} csv files in {exp_path}")

        for csv_file in csv_files:
            csv_path = os.path.join(exp_path, csv_file)
            print(f"Reading CSV file: {csv_file}")
            df = pd.read_csv(csv_path)
            df["model"] = exp_subdir
            df_all_list.append(df)

    if len(df_all_list) > 0:
        df_all = pd.concat(df_all_list, ignore_index=True)
        print(f"\nConcatenating {len(df_all_list)} dataframes into df_all...")
        print(f"df_all shape: {df_all.shape}")
    else:
        df_all = pd.DataFrame()


    if one_hot:
        ohe_experiment_dirs = [e for e in experiment_dirs if e == "one_hot"][0]
        ohe_df_all_list = []
        ohe_dir = os.path.join(embedding_base_dir, ohe_experiment_dirs)
        csv_files = [f for f in os.listdir(ohe_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(ohe_dir, csv_file))
            ohe_df_all_list.append(df)
        if len(ohe_df_all_list) > 0:
            ohe_df_all = pd.concat(ohe_df_all_list, ignore_index=True)
        else:
            ohe_df_all = pd.DataFrame()

        ohe_df_all["model_name"] = "ohe"

        return (df_all, ohe_df_all)

    return df_all

def evaluate_classifier(score, 
                        predicted_label, 
                        gt_label,
                        label_true=0,
                        label_false=1):
    tp = sum((predicted_label == label_true) & (gt_label == label_true))
    tn = sum((predicted_label == label_false) & (gt_label == label_false))
    fp = sum((predicted_label == label_true) & (gt_label == label_false))
    fn = sum((predicted_label == label_false) & (gt_label == label_true))
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = np.sum(predicted_label == gt_label) / len(predicted_label)
    roc = roc_auc_score(gt_label, score)

    # Calculate the precision-recall AUC
    precision_values, recall_values, _ = precision_recall_curve(gt_label, score)
    pr_auc = auc(recall_values, precision_values)

    ordered_scores = np.argsort(score)[0:100]
    top_100_pct = sum(gt_label[ordered_scores] == label_true) / 100
    evaluation = {
        "tp" : tp,
        "tn" : tn,
        "fp" : fp,
        "fn" : fn,
        'pr_auc' : pr_auc,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1,
        "accuracy" : accuracy,
        "roc" : roc,
        "top_100_pct": top_100_pct
    }
    return evaluation


def load_df_with_budget(base_path):
    # List of model names you want to include
    model_names = [
        "esm_8m",
        "esm_35m",
        "esm_150m",
        "esm_650m",
        "esm_3b",
        "one_hot",
        "progen2-small",
        "progen2-medium",
        "prot_bert"
    ]

    dfs = []
    for model_name in model_names:
        # Glob can handle slight filename variations (if needed)
        pattern = f"{base_path}{model_name}"
        matching_files = glob.glob(pattern)
        if not matching_files:
            # Optionally, print or warn
            print(f"Warning: No file found for {model_name} with pattern {pattern}")
            continue
        # We expect only one match per model_name
        df_temp = pd.read_csv(matching_files[0])
        df_temp["model_name"] = model_name
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)

    sets = df["set_name"].unique()
    return (dict([(set_name, df[df["set_name"] == set_name]) for set_name in sets]))

def get_one_hot_encoding(sdf, first_col, last_col):
    si = np.where(sdf.columns == first_col)[0][0]
    ei = np.where(sdf.columns == last_col)[0][0]
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[sdf.columns[si:(ei+1)]]).to_numpy()).to(torch.int64)
    return one_hot_encoding


def get_positions_gfp_protgym(df, first_col, last_col):
    si = np.where(df.columns == first_col)[0][0]
    ei = np.where(df.columns == last_col)[0][0]+1
    return [int(pos[1:]) for pos in df.columns[si:ei].to_list()]

def get_relevant_columns_gfp_protgym(df, first_col, last_col):
    si = np.where(df.columns == first_col)[0][0]
    ei = np.where(df.columns == last_col)[0][0]+1
    return df.columns[si:ei]


def gcn4_positions(df):
    return get_positions_gfp_protgym(df, positions["gcn4"][0], positions["gcn4"][1])

def gcn4_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["gcn4"][0], positions["gcn4"][1])    

def pard3_positions(df):
    return get_positions_gfp_protgym(df, positions["pard3"][0], positions["pard3"][1])

def pard3_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["pard3"][0], positions["pard3"][1])    

def lov_positions(df):
    return get_positions_gfp_protgym(df, positions["lov"][0], positions["lov"][1])

def lov_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["lov"][0], positions["lov"][1])    

def gfp_positions(df):
    return get_positions_gfp_protgym(df, positions["gfp"][0], positions["gfp"][1])

def gfp_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["gfp"][0], positions["gfp"][1])

def pte_positions(df):
    return get_positions_gfp_protgym(df, positions["pte"][0], positions["pte"][1])

def pte_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["pte"][0], positions["pte"][1])

def aamyl_positions(df):
    return get_positions_gfp_protgym(df, positions["aamyl"][0], positions["aamyl"][1])

def aamyl_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["aamyl"][0], positions["aamyl"][1])

def his_positions(df):
    return get_positions_gfp_protgym(df, positions["his"][0], positions["his"][1])

def his_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["his"][0], positions["his"][1])

def his2_positions(df):
    return get_positions_gfp_protgym(df, positions["his2"][0], positions["his2"][1])

def his2_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["his2"][0], positions["his2"][1])

def his5_positions(df):
    return get_positions_gfp_protgym(df, positions["his5"][0], positions["his5"][1])

def his5_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["his5"][0], positions["his5"][1])

def casp_positions(df):
    return get_positions_gfp_protgym(df, positions["casp"][0], positions["casp"][1])

def casp_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["casp"][0], positions["casp"][1])

def nmt_positions(df):
    si = np.where(df.columns == "1")[0][0]
    ei = np.where(df.columns == "272")[0][0]+1
    positions_with_mutations =  np.array([len(pd.unique(df.iloc[:,i])) > 1 for i in range(si,ei)])
    return df.columns[si:ei][positions_with_mutations].to_list()

def nmt_relevant_columns(df):
    si = np.where(df.columns == "1")[0][0]
    ei = np.where(df.columns == "272")[0][0]+1
    positions_with_mutations =  np.array([len(pd.unique(df.iloc[:,i])) > 1 for i in range(si,ei)])
    return df.columns[si:ei][positions_with_mutations]

get_positions = {
    "gcn4": gcn4_positions,
    "pard3": pard3_positions,
    "lov": lov_positions,
    "gfp": gfp_positions,
    "pte": pte_positions,   
    "nmt": nmt_positions,
    "aamyl": aamyl_positions,
    "his": his_positions,
    "his2": his2_positions,
    "his5": his5_positions,
    "casp": casp_positions
}

get_relevant_columns = {
    "gcn4": gcn4_relevant_columns,
    "pard3": pard3_relevant_columns,
    "lov": lov_relevant_columns,
    "gfp": gfp_relevant_columns,
    "pte": pte_relevant_columns,   # 
    "nmt": nmt_relevant_columns,
    "aamyl": aamyl_relevant_columns,
    "his": his_relevant_columns,
    "his2": his2_relevant_columns,
    "his5": his5_relevant_columns,
    "casp": casp_relevant_columns
}


def round_up_trim(val, decimals=0):
    """
    Rounds a number up to a specified number of decimal places, and trims trailing zeros.
    
    Args:
        val (float or int): The number to round.
        decimals (int): Number of decimal places to round up to.
    
    Returns:
        str: The rounded number as string without unnecessary trailing zeros or decimal points.
    """
    from math import ceil, pow, isclose

    # If val is int, just return as str
    if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
        return str(int(val))

    # Ensure it's float
    fval = float(val)
    multiplier = pow(10, decimals)
    rounded_up = ceil(fval * multiplier) / multiplier

    # Format with enough decimals, then strip trailing zeros and possible trailing decimal point
    fmt_str = "{:." + str(decimals) + "f}"
    result = fmt_str.format(rounded_up).rstrip('0').rstrip('.')
    # edge-case: .0
    if result == '':
        result = '0'
    return result

def fix_ticks(ax, 
              x=True, 
              y=True, 
              every_other=True, 
              fontsize=8.5,
              xlim=None,
              ylim=None,
              xbreaks=None,
              ybreaks=None,
              decimals=2,
              verbose=False):
    if x:

        if xlim is not None and xbreaks is not None:
            x_ticks = np.linspace(xlim[0], xlim[1], xbreaks)

        elif xlim is not None:
            x_ticks = ax.get_xticks()
            x_ticks = [x for x in x_ticks if float(x) >= xlim[0] and float(x) <= xlim[1]]
        else:
            x_ticks = ax.get_xticks()
            

        if len(x_ticks) > 0 and len(x_ticks) <= 3:
            every_other = False

        if every_other:
            x_ticks_labels = [round_up_trim(x, decimals=decimals) if i % 2 == 0 else "" for i, x in enumerate(x_ticks)]
        else:
            pos = [0, len(x_ticks) - 1]
            x_ticks_labels = [round_up_trim(x, decimals=decimals) if i in pos else "" for i, x in enumerate(x_ticks)]

        if verbose:
            print("SELECTED X TICKS:")
            print(x_ticks)
            print("SELECTED X LABELS:")
            print(x_ticks_labels)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels, fontsize=fontsize)

    if y:
        if ylim is not None and ybreaks is not None:
            y_ticks = np.linspace(ylim[0], ylim[1], ybreaks)
        elif ylim is not None:
            y_ticks = ax.get_yticks()
            y_ticks = [y for y in y_ticks if float(y) >= ylim[0] and float(y) <= ylim[1]]
        else:
            y_ticks = ax.get_yticks()

        if len(y_ticks) > 0 and len(y_ticks) <= 3:
            every_other = False

        if every_other:
            y_ticks_labels = [round_up_trim(y, decimals=decimals) if i % 2 == 0 else "" for i, y in enumerate(y_ticks)]
        else:
            pos = [0, len(y_ticks) - 1]
            y_ticks_labels = [round_up_trim(y, decimals=decimals) if i in pos else "" for i, y in enumerate(y_ticks)]
            
        if verbose:
            print("SELECTED Y TICKS:")
            print(y_ticks)
            print("SELECTED Y LABELS:")
            print(y_ticks_labels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_labels, fontsize=fontsize)
        
def has_point_zero(f):
    """Return True if the float has exactly .0 after the decimal point (e.g., 2.0 -> True, 2.1 -> False)"""
    return isinstance(f, float) and f == int(f)


def get_labels(df_name, discrete=False, df=None):
    if df is None:
        df = pd.read_csv(DATASET_PATHS[df_name])

    if df_name == "gfp":
        return((df["inactive"] == False).astype(int).to_numpy())

    activity_col_name = "p-nitrophenyl_octanoate" if df_name == "pte" else "activity"
    labels = df[activity_col_name].astype(float).to_numpy()

    if discrete:
        return (labels > np.median(labels)).astype(int)

    return labels


# def plot_exporter(
#     output_path,
#     default_format="svg",
#     dpi=300,
#     transparent=False,
#     pad_inches=0.02,
# ):
#     """Create a plot exporter using Helvetica with sans-serif fallbacks."""
#     output_path = Path(output_path)
#     output_path.mkdir(parents=True, exist_ok=True)

#     # Preserve editable text in SVG and TrueType fonts in PDF.
#     plt.rcParams["svg.fonttype"] = "none"
#     plt.rcParams["pdf.fonttype"] = 42
#     plt.rcParams["ps.fonttype"] = 42

#     # Prefer Helvetica throughout the figure.
#     plt.rcParams["font.family"] = "sans-serif"
#     plt.rcParams["font.sans-serif"] = [
#         "Helvetica",
#         "Arial",
#         "DejaVu Sans",
#     ]

#     # Use sans-serif math text.
#     plt.rcParams["mathtext.fontset"] = "dejavusans"
#     plt.rcParams["mathtext.default"] = "regular"

#     def export_plot(name, fig=None, file_format=None, close=False, **savefig_kwargs):
#         fig = plt.gcf() if fig is None else fig
#         name = Path(name)

#         if file_format is None and name.suffix:
#             file_format = name.suffix.lstrip(".")
#         elif file_format is None:
#             file_format = default_format

#         file_format = file_format.lower().lstrip(".")
#         save_path = output_path / f"{name.stem}.{file_format}"

#         save_options = {
#             "format": file_format,
#             "dpi": dpi,
#             "bbox_inches": "tight",
#             "pad_inches": pad_inches,
#             "transparent": transparent,
#             "facecolor": "none" if transparent else "white",
#         }

#         save_options.update(savefig_kwargs)
#         fig.savefig(save_path, **save_options)

#         if close:
#             plt.close(fig)

#         return save_path

#     return export_plot

def plot_exporter(
    output_path,
    default_format="svg",
    dpi=300,
    transparent=False,
    pad_inches=0.02,
    style_axes=False,
    axis_linewidth=0.5,
    axis_color="#58595b",
):
    """Create a plot exporter using Helvetica with sans-serif fallbacks."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Preserve editable text in SVG and TrueType fonts in PDF.
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Prefer Helvetica throughout the figure.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    # Use sans-serif math text.
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"

    def apply_axis_style(fig):
        for ax in fig.axes:
            for spine in ax.spines.values():
                spine.set_linewidth(axis_linewidth)
                spine.set_color(axis_color)

            ax.tick_params(
                axis="both",
                which="both",
                width=axis_linewidth,
                color=axis_color,
                #labelcolor=axis_color,
            )

            # ax.xaxis.label.set_color(axis_color)
            # ax.yaxis.label.set_color(axis_color)

    def export_plot(name, fig=None, file_format=None, close=False, **savefig_kwargs):
        fig = plt.gcf() if fig is None else fig
        name = Path(name)

        if file_format is None and name.suffix:
            file_format = name.suffix.lstrip(".")
        elif file_format is None:
            file_format = default_format

        file_format = file_format.lower().lstrip(".")
        save_path = output_path / f"{name.stem}.{file_format}"

        if style_axes:
            apply_axis_style(fig)

        save_options = {
            "format": file_format,
            "dpi": dpi,
            "bbox_inches": "tight",
            "pad_inches": pad_inches,
            "transparent": transparent,
            "facecolor": "none" if transparent else "white",
        }
        save_options.update(savefig_kwargs)

        fig.savefig(save_path, **save_options)

        if close:
            plt.close(fig)

        return save_path

    return export_plot

def fitness_from_prob(pssm, wt_tensor, variant_tensor):    
    wt_one_hot = torch.nn.functional.one_hot(torch.tensor(wt_tensor), pssm.shape[1])
    variant_one_hot = torch.nn.functional.one_hot(torch.tensor(variant_tensor), pssm.shape[1])   
    # return (torch.mean(torch.log(torch.sum(pos_prob_matrix * variant_one_hot, dim=1)) - torch.log(torch.sum(pos_prob_matrix * wt_one_hot, dim=1))))
    # B for batch size, S for number of mutated positions, V for vocabulary size
    #return torch.mean((torch.log(torch.einsum("BSV,SV->BSV", variant_one_hot[:,:,:].to(torch.float), pssm)) - torch.log(torch.sum(pssm * wt_one_hot, dim=1))))
    return torch.log(pssm[:,variant_tensor]).view((-1))  - torch.log(pssm[:,wt_tensor]).view((-1))

def fitness_from_prob_non_dms(pssm, wt_tensor, variant_tensor):
    wt_one_hot = torch.nn.functional.one_hot(torch.tensor(wt_tensor), pssm.shape[1])
    variant_one_hot = torch.nn.functional.one_hot(torch.tensor(variant_tensor), pssm.shape[1])   
    
    # B for batch size, S for number of mutated positions, V for vocabulary size
    return torch.mean((torch.log(torch.einsum("BSV,SV->BS", 
                                              variant_one_hot[:,:,:].to(torch.float), 
                                              pssm)) -
                       torch.log(torch.sum(pssm * wt_one_hot, dim=1))),
                      dim=1)

def fitness_from_prob_non_dms(pssm, wt_tensor, variant_tensor, device=torch.device("cpu")):
    # wt_one_hot = torch.nn.functional.one_hot(torch.tensor(wt_tensor), pssm.shape[1]).to(device)
    # variant_one_hot = torch.nn.functional.one_hot(torch.tensor(variant_tensor), pssm.shape[1]).to(device)
    wt_one_hot = torch.nn.functional.one_hot(wt_tensor.clone().detach(), pssm.shape[1]).to(device)
    variant_one_hot = torch.nn.functional.one_hot(variant_tensor.clone().detach(), pssm.shape[1]).to(device)

    
    # B for batch size, S for number of mutated positions, V for vocabulary size
    return torch.mean((torch.log(torch.einsum("BSV,SV->BS", 
                                              variant_one_hot[:,:,:].to(torch.float), 
                                              pssm)) -
                       torch.log(torch.sum(pssm * wt_one_hot, dim=1))),
                      dim=1)

def split_train_test():
    pass
    

# Mutation data_tensor should be of shape (N, 3, S)
# where N is the overall number of sequences, 
# and S is each sequence length.
#
# In this tensors there are three (N,S) tensors stacked at the 2nd dimension (dim=1)
# The 3rd (idx=2) tensor is a one_hot encoding  describing the identity of the mutations in the sequence
# The 2nd (idx=1) tensor is a one_hot encoding where instead one's are replaced by the identity (in the vocabulary) of the new AA (mutation identity)
# The 1nd (idx=0) tensor is a one_hot encoding where instead one's are replaced by the identity (in the vocabulary) of the old AA (original identity)
# 
# Fitness tensor is shaped by (N, 1) and describes the fitness of each sequence

def dms_create_contrastive_pairs(mutation_data_tensor, fitness_tensor):
    mutation_positions = torch.where(mutation_data_tensor[:,2,:] > 0)[1]
    unique_positions = torch.unique(mutation_positions)
    
    all_pairs = torch.tensor([], dtype=torch.int)
    
    for position in unique_positions:
        sequences_mutated_in_position = torch.where(mutation_positions == position)[0]
        fitness_of_sequences_in_position = fitness_tensor[sequences_mutated_in_position]
        median_fitness = torch.median(fitness_of_sequences_in_position)
        negative_sequences = sequences_mutated_in_position[torch.where(fitness_of_sequences_in_position <= median_fitness)]
        positive_sequences = sequences_mutated_in_position[torch.where(fitness_of_sequences_in_position > median_fitness)]
        
        concat_pairs = \
        torch.cat([torch.stack([torch.stack([negative,positive]) for positive in positive_sequences]) for negative in negative_sequences])
        
        all_pairs = torch.cat([all_pairs, concat_pairs])

def dms_get_working_positions(dms_df):
    return(np.unique([int(mt[1:3]) for mt in dms_df["mutant"].to_list()]))


def align_sequences(misaligned_seq):
    copy_indices = []
    shift_counter = 0
    copied_counter = 0
            
            
    for i,s in enumerate(misaligned_seq):
        if s == "-":
            shift_counter += 1
        else:
            copy_indices.append(copied_counter + shift_counter + 1)
            copied_counter += 1
            
    return copy_indices, shift_counter, copied_counter



# Has to return a list of from, to, pos (PDB pos - one based index!!!)
def get_mutated_position_full_mask_by_first_last_colname(sequence_df, first_col, last_col, sequence_col):
    
    si = np.where(sequence_df.columns == first_col)[0][0]
    ei = np.where(sequence_df.columns == last_col)[0][0] + 1
    pos = [int(x[1:]) for x in sequence_df.columns[si:ei].to_list()]
    from_mut = [x[0] for x in sequence_df.columns[si:ei].to_list()]
    to_mut = sequence_df.iloc[:,si:ei].values.tolist()#[list(x) for x in sequence_df[sequence_col].to_list()]
    
    return [from_mut], to_mut, [pos]



# Has to return a list of from, to, pos (PDB pos - one based index!!!)å
def get_mutated_position_partial_mask_by_first_last_colname(sequence_df, first_col, last_col, sequence_col):
    
    si = np.where(sequence_df.columns == first_col)[0][0]
    ei = np.where(sequence_df.columns == last_col)[0][0] + 1
    pos = [int(x[1:]) for x in sequence_df.columns[si:ei].to_list()]
    from_mut = [x[0] for x in sequence_df.columns[si:ei].to_list()]
    
    # to_mut = [[aa for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df[sequence_col].to_list()]
    # pos_var = [[pos[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df[sequence_col].to_list()]
    # from_mut_var = [[from_mut[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df[sequence_col].to_list()]
    
    used_seqs = sequence_df.iloc[:,si:ei].values.tolist()
    to_mut = [[aa for i,aa in enumerate(x) if aa != from_mut[i]] for x in used_seqs]
    pos_var = [[pos[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in used_seqs]
    from_mut_var = [[from_mut[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in used_seqs]
    
    
    return from_mut_var, to_mut, pos_var


# Has to return a list of from, to, pos (PDB pos - one based index!!!)
def get_mutated_position_function_gfp(sequence_df):
    
    si = np.where(sequence_df.columns == 'L42')[0][0]
    ei = np.where(sequence_df.columns == 'V224')[0][0] + 1
    pos = [int(x[1:]) for x in sequence_df.columns[si:ei].to_list()]
    from_mut = [x[0] for x in sequence_df.columns[si:ei].to_list()]
    to_mut = [list(x) for x in sequence_df.sequence.to_list()]
    
    return [from_mut], to_mut, [pos]



# Has to return a list of from, to, pos (PDB pos - one based index!!!)
def get_mutated_position_function_gfp_n2(sequence_df):
    
    si = np.where(sequence_df.columns == 'L42')[0][0]
    ei = np.where(sequence_df.columns == 'V224')[0][0] + 1
    pos = [int(x[1:]) for x in sequence_df.columns[si:ei].to_list()]
    from_mut = [x[0] for x in sequence_df.columns[si:ei].to_list()]
    
    to_mut = [[aa for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df.sequence.to_list()]
    pos_var = [[pos[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df.sequence.to_list()]
    from_mut_var = [[from_mut[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df.sequence.to_list()]
    
    return from_mut_var, to_mut, pos_var


def plot_hists(active, inactive, save_path=None):
    plt.hist(active, bins=30, alpha=0.6, label='Active', color='green', density=True)
    plt.hist(inactive, bins=30, alpha=0.6, label='Inactive', color='gray', density=True)
        
    plt.title("Overlaid Histograms [Separation: %.4f]" % float(np.median(active) - np.median(inactive)))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show(block=False)
    plt.pause(8)
    #time.sleep(8)
    plt.close()


def get_one_hot_encoding(sdf, first_col, last_col):
    si = np.where(sdf.columns == first_col)[0][0]
    ei = np.where(sdf.columns == last_col)[0][0]
    
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[sdf.columns[si:(ei+1)]]).to_numpy()).to(torch.int64)

    return(one_hot_encoding)



deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)
    
def read_sequence(filename: str) -> tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> list[tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

def get_non_alignment_regions(s):
    regions = []
    pad_region = False
    aa_start = None

    for i, c in enumerate(s):
        if c != '-':
            if not pad_region:
                aa_start = i
                pad_region = True
        else:
            if pad_region:
                aa_end = i - 1
                regions.append((aa_start, aa_end))
                pad_region = False
    if pad_region:
        regions.append((aa_start, len(s) - 1))
    # Now, format as "start_end" for first, then "xstartxend" for others
    if not regions:
        return ""
    out = []
    for idx, (st, en) in enumerate(regions):
            out.append(f"{st}_{en}")

    return "x".join(out)

# Example usage:
# s = "---XBC--X--A--XXX--"
# print(get_non_alignment_regions(s))  # Output: "3_5x8x8x11x11x14_16"

    
def generate_msa_df_from_aligned_msa(aligned_msa_path):
    msa = read_msa(aligned_msa_path)
    msa_str = [msa[i][1] for i in range(0, len(msa))]
    non_pad_regions = [get_non_alignment_regions(s) for s in msa_str]
    msa_df = pd.DataFrame({"sequence": msa_str,
                           "pad_regions": non_pad_regions})
    return msa_df
    

def pairwise_cosine(X):
    X = F.normalize(X, dim=-1)
    similarity = torch.matmul(X, X.t())     # [N, N]
    distance = 1 - similarity
    return distance

def online_mine_triplets(labels):
    
    triplets = []
    
    for i, anchor_label in enumerate(labels):        
        positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
        negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]

        for pos_idx in positive_indices:
            if pos_idx == i: continue
            for neg_idx in negative_indices:
                triplets.append((i, pos_idx.item(), neg_idx.item()))
                
    return triplets