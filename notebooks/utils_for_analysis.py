# Plotting variance explained bars (following generate_budgeted_subsamples style)
import numpy as np
import pandas as pd
import os
import glob
import torch


from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc
)


DATASET_PATHS = {
    "gfp" : "data/gfp/gfp_dataset_10mut.csv",
    "lov": "data/lov/lov.csv",
    "pard3": "data/pard3/pard3.csv",
    "gcn4": "data/gcn4/gcn4.csv",
    "pte": "data/pte/pte.csv",
    "nmt": "data/nmt/nmt.csv",
    "aamyl": "data/aamyl/aamyl.csv",
    "trpb": "data/trpb/trpb.csv",
    "his": "data/his/his.csv",
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
    "his": ["L7", "D211"],
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
    "his": "full_seq",
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
    "his": "num_muts",
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
        pattern = f"{base_path}mlp_llm_200_20_{model_name}"
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

def lov_relevant_columns():
    return get_relevant_columns_gfp_protgym(df, positions["lov"][0], positions["lov"][1])    

def gfp_positions(df):
    return get_positions_gfp_protgym(df, positions["gfp"][0], positions["gfp"][1])

def gfp_relevant_columns():
    return get_relevant_columns_gfp_protgym(df, positions["gfp"][0], positions["gfp"][1])

def pte_positions(df):
    return get_positions_gfp_protgym(df, positions["pte"][0], positions["pte"][1])

def pte_relevant_columns():
    return get_relevant_columns_gfp_protgym(df, positions["pte"][0], positions["pte"][1])

def aamyl_positions(df):
    return get_positions_gfp_protgym(df, positions["aamyl"][0], positions["aamyl"][1])

def aamyl_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["aamyl"][0], positions["aamyl"][1])

def his_positions(df):
    return get_positions_gfp_protgym(df, positions["his"][0], positions["his"][1])

def his_relevant_columns(df):
    return get_relevant_columns_gfp_protgym(df, positions["his"][0], positions["his"][1])

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


