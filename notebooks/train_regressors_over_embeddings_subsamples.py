import argparse
import pandas as pd
import numpy as np
import os
import torch

from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

DATASET_PATHS = {
    "gfp" : "data/gfp/gfp_dataset_10mut_nmut_1.csv",
    "lov": "data/lov/lov.csv",
    "pard3": "data/pard3/pard3.csv",
    "gcn4": "data/gcn4/gcn4.csv",
    "pte": "data/pte/pte.csv",
    "nmt": "data/nmt/nmt_full_seq.csv",
}

positions = {
    "gfp": ["L42", "V224"],
    "lov": ["G2", "T112"],
    "pard3": ["L48","R82"],
    "gcn4": ["S101","S144"],
    "pte": ["I72", "M283"],
    "nmt": ["1", "272"]
}

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
    if dataset == "nmt":
        return nmt_relevant_columns(df)
    elif dataset in positions:
        first, last = positions[dataset][0], positions[dataset][1]
        cols = get_relevant_columns_gfp_protgym(df, first, last)
        return cols
    else:
        raise ValueError(f"Unknown dataset for OHE column range: {dataset}")

def get_label_column(dataset, df):
    if dataset == "gfp":
        return df["activity"].values
    elif dataset == "nmt":
        return df["activity"].values
    elif dataset in ["pard3", "lov", "gcn4", "pte"]:
        if "fitness" in df.columns:
            return df["fitness"].values
        elif "activity" in df.columns:
            return df["activity"].values
        else:
            raise ValueError(f"Cannot find label column for {dataset}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_one_hot_encoding(sdf, relevant_columns):
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[relevant_columns]).to_numpy()).to(torch.float32)
    return one_hot_encoding

def get_embedding_paths(dataset, base_path):
    # Ensure all datasets support progen2-small embedding
    if dataset == "nmt":
        return {
            "progen2-small": f"{base_path}/data/nmt/embeddings/progen2-small",
            "esm_8m": f"{base_path}/data/nmt/embeddings/esm_8m"
        }
    elif dataset == "gfp":
        return {
            "esm_8m": f"{base_path}/data/gfp/embeddings/esm_8m",
            "progen2-small": f"{base_path}/data/gfp/embeddings/progen2-small"
        }
    elif dataset == "pard3":
        return {
            "esm_8m": f"{base_path}/data/pard3/embeddings/esm_8m",
            "progen2-small": f"{base_path}/data/pard3/embeddings/progen2-small"
        }
    elif dataset == "lov":
        return {
            "esm_8m": f"{base_path}/data/lov/embeddings/esm_8m",
            "progen2-small": f"{base_path}/data/lov/embeddings/progen2-small"
        }
    elif dataset == "pte":
        return {
            "esm_8m": f"{base_path}/data/pte/embeddings/esm_8m",
            "progen2-small": f"{base_path}/data/pte/embeddings/progen2-small"
        }
    elif dataset == "gcn4":
        return {
            "esm_8m": f"{base_path}/data/gcn4/embeddings/esm_8m",
            "progen2-small": f"{base_path}/data/gcn4/embeddings/progen2-small"
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="nmt", help='Dataset name')
    parser.add_argument('--base_path', type=str, default="/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/", help='Base path')
    parser.add_argument('--output_csv', type=str, default=None, help='Where to save results')
    parser.add_argument('--iters', type=int, default=30, help='Number of iterations for each sample size')
    parser.add_argument('--n_samples', type=int, nargs='+', default=[5, 10, 25, 50, 75, 100, 125, 150, 200, 250, 500, 1000], help='Training sample sizes per experiment')
    parser.add_argument('--shuffle_labels', action='store_true', default=False, help='Shuffle labels for control analysis')
    parser.add_argument('--ohe_hl', type=int, nargs='+', default=[32], help='Hidden layer sizes for OHE MLP')
    parser.add_argument('--llm_hl', type=int, nargs='+', default=[200, 20], help='Hidden layer sizes for LLM MLP')
    parser.add_argument('--ohe_solver', type=str, default='lbfgs', choices=['adam', 'lbfgs'], help='Solver for OHE MLP')
    parser.add_argument('--llm_solver', type=str, default='lbfgs', choices=['adam', 'lbfgs'], help='Solver for LLM MLP')
    parser.add_argument('--ohe_alpha', type=float, default=1e-4, help='Alpha for OHE MLP')
    parser.add_argument('--llm_alpha', type=float, default=1e-3, help='Alpha for LLM MLP')
    parser.add_argument('--ohe_learning_rate_init', type=float, default=2e-4, help='Learning rate for OHE MLP')
    parser.add_argument('--llm_learning_rate_init', type=float, default=1e-3, help='Learning rate for LLM MLP')
    parser.add_argument('--ohe_max_iter', type=int, default=100, help='Max iterations for OHE MLP')
    parser.add_argument('--llm_max_iter', type=int, default=100, help='Max iterations for LLM MLP')
    parser.add_argument('--model_names', nargs='+', default=None, help='Limit to subset of embedding models')
    parser.add_argument('--mean_embeddings', action='store_true', default=False,
                        help='If set, use mean embedding vectors (i.e., take mean on axis=1, not flatten)')
    # Add support for external labels (column from df, like in train_classifiers_over_embeddings.py)
    parser.add_argument('--external_labels_column', type=str, default=None,
                        help="If set, use this column from the dataframe as labels for all regression instead of the default activity/fitness column or embedding labels.")
    args = parser.parse_args()

    base_path = args.base_path
    dataset = args.dataset_name
    df = pd.read_csv(DATASET_PATHS[dataset])
    relevant_columns = get_relevant_columns(dataset, df)
    one_hot = get_one_hot_encoding(df, relevant_columns)
    assert one_hot.shape[1] == sum([len(pd.unique(df[C])) for C in relevant_columns])

    # Add sanity print if using external labels
    if args.external_labels_column is not None:
        labels = torch.tensor(df[args.external_labels_column].values).float()
        print(f"Using external labels column: {args.external_labels_column}")
        print("Sanity check: a few external labels:", labels[:8].tolist())
    else:
        labels = torch.tensor(get_label_column(dataset, df)).float()
    original_labels = labels.clone()
    embedding_paths = get_embedding_paths(dataset, base_path)
    if args.model_names is not None:
        embedding_paths = {k: v for k, v in embedding_paths.items() if k in args.model_names}

    labels_all = {}
    indices_all = {}
    embeddings_all = {}
    external_labels_all = {}
    for model_name, model_path in embedding_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Embedding path {model_path} does not exist.")
        labels_emb = torch.load(os.path.join(model_path, "y_values.pt"))
        indices_emb = torch.load(os.path.join(model_path, "indices.pt"))
        embeddings = torch.load(os.path.join(model_path, "embeddings.pt"))

        # Apply mean or flatten, depending on flag
        if args.mean_embeddings:
            emb_proc = embeddings.mean(axis=1)
        else:
            emb_proc = embeddings.reshape(embeddings.shape[0], -1)

        # Normalize embeddings
        normalized_embeddings = emb_proc - emb_proc.mean(dim=0, keepdim=True)
        normalized_embeddings = normalized_embeddings / emb_proc.std(dim=0, keepdim=True)
        labels_all[model_name] = labels_emb
        indices_all[model_name] = indices_emb
        embeddings_all[model_name] = normalized_embeddings

        # If args.external_labels_column is set, we need to map external labels to embedding indices
        if args.external_labels_column is not None:
            _ext_labels = torch.tensor(df.iloc[indices_emb][args.external_labels_column].values).float()
            external_labels_all[model_name] = _ext_labels
            print(f"Sanity check on external labels for {model_name}:",
                  _ext_labels[:8].tolist())

    ohe_mlp_params = {
        "activation": 'relu',
        "solver": args.ohe_solver,
        "alpha": args.ohe_alpha,
        "learning_rate_init": args.ohe_learning_rate_init,
        "max_iter": args.ohe_max_iter,
        "random_state": 4321,
        "n_iter_no_change": 10,
        "verbose": False
    }

    llm_mlp_params = {
        "activation": 'relu',
        "solver": args.llm_solver,
        "alpha": args.llm_alpha,
        "learning_rate_init": args.llm_learning_rate_init,
        "max_iter": args.llm_max_iter,
        "random_state": 4321,
        "n_iter_no_change": 10,
        "verbose": False
    }

    all_results = []
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = f"data/{dataset}/regression_result_by_training_samples.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Always run both (regular + shuffled) if shuffle_labels is set
    shuffle_modes = [False, True] if args.shuffle_labels else [False]

    for N_samples in args.n_samples:
        if N_samples > int(one_hot.shape[0] * 0.7):
            continue
        for iter in range(args.iters):

            train_indices = np.random.choice(one_hot.shape[0], N_samples, replace=False)
            test_indices = np.setdiff1d(np.arange(one_hot.shape[0]), train_indices)
            result_dict = {"N_samples": N_samples, "iter": iter}

            for shuffle in shuffle_modes:
                if shuffle:
                    prefix = "shuffled_"
                    curr_labels = original_labels[torch.randperm(original_labels.size(0))]
                else:
                    prefix = ""
                    curr_labels = original_labels

                print("Fitting OHE MLP on %dx%d, %d labels %s" % (one_hot.shape[0], one_hot.shape[1], len(curr_labels), "(shuffled)" if shuffle else ""))
                mlp_ohe = MLPRegressor(hidden_layer_sizes=tuple(args.ohe_hl), **ohe_mlp_params)
                mlp_ohe.fit(one_hot.numpy()[train_indices], curr_labels.numpy()[train_indices])
                cor_ohe = spearmanr(mlp_ohe.predict(one_hot.numpy()[test_indices]), curr_labels.numpy()[test_indices])
                result_dict[f"{prefix}cor_ohe"] = cor_ohe.correlation

                for model_name, model_path in embedding_paths.items():
                    normalized_embeddings = embeddings_all[model_name]
                    if args.external_labels_column is not None:
                        # Use mapped external labels
                        curr_llm_labels = external_labels_all[model_name]
                    else:
                        # Use embedding's provided y_values.pt (already subsetted)
                        curr_llm_labels = labels_all[model_name]

                    if shuffle:
                        curr_llm_labels = curr_llm_labels[torch.randperm(curr_llm_labels.size(0))]

                    # Map one_hot indices to embedding indices for training/test splits
                    emb_indices = indices_all[model_name].numpy()
                    emb_train_mask = np.isin(emb_indices, train_indices)
                    emb_test_mask = np.isin(emb_indices, test_indices)
                    emb_train_indices = np.where(emb_train_mask)[0]
                    emb_test_indices = np.where(emb_test_mask)[0]

                    print("Fitting %s embeddings MLP on %dx%d, %d labels %s" %
                          (model_name, normalized_embeddings.shape[0], normalized_embeddings.shape[1], len(curr_llm_labels), "(shuffled)" if shuffle else ""))

                    mlp_llm = MLPRegressor(hidden_layer_sizes=tuple(args.llm_hl), **llm_mlp_params)
                    mlp_llm.fit(
                        normalized_embeddings.numpy()[emb_train_indices],
                        curr_llm_labels.numpy()[emb_train_indices]
                    )
                    pred_llm = mlp_llm.predict(normalized_embeddings.numpy()[emb_test_indices])
                    test_llm_labels = curr_llm_labels.numpy()[emb_test_indices]
                    cor_llm = spearmanr(pred_llm, test_llm_labels)

                    # Record whether mean or flat
                    result_key = f"{prefix}cor_{model_name}"
                    if args.mean_embeddings:
                        result_key += "_mean"
                    else:
                        result_key += "_flat"
                    result_dict[result_key] = cor_llm.correlation
            all_results.append(result_dict)
            print(result_dict)
            # Save every 5th iteration in the loop, but only for the last shuffle mode in the list (to avoid duplicate saves per iter)
            if (iter + 1) % 5 == 0 and shuffle == shuffle_modes[-1]:
                pd.DataFrame(all_results).to_csv(output_csv, index=False)
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(output_csv, index=False)
    print(result_df)

if __name__ == '__main__':
    main()

# Example run:
# python train_regressors_over_embeddings_subsamples.py --dataset_name nmt --base_path /home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/ --iters 10 --n_samples 1 5 10 25 --llm_hl 200 20 --ohe_hl 32 --output_csv outputs/nmt_regression_results.csv
# Add --mean_embeddings for mean pooling:
# python train_regressors_over_embeddings_subsamples.py ... --mean_embeddings