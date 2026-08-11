import argparse
import pandas as pd
import numpy as np
import os
import torch

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
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
        if "fitness" in df.columns:
            return df["fitness"].values
        if "activity" in df.columns:
            return df["activity"].values
        raise ValueError(f"Unknown dataset and no fitness/activity column found: {dataset}")

def get_one_hot_encoding(sdf, relevant_columns):
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[relevant_columns]).to_numpy()).to(torch.float32)
    return one_hot_encoding

def get_dataset_path(dataset, base_path, dataset_path=None):
    if dataset_path is not None:
        return dataset_path
    if dataset in DATASET_PATHS:
        return DATASET_PATHS[dataset]
    return os.path.join(base_path, "data", dataset, f"{dataset}.csv")

def get_relevant_columns_from_args(dataset, df, first_col=None, last_col=None):
    if first_col is not None or last_col is not None:
        if first_col is None or last_col is None:
            raise ValueError("Provide both --first_col and --last_col, or neither.")
        return get_relevant_columns_gfp_protgym(df, first_col, last_col)
    if dataset in positions or dataset == "nmt":
        return get_relevant_columns(dataset, df)
    stop_candidates = [col for col in ["full_seq", "full_sequence", "sanity_mut_numb", "activity", "fitness", "num_muts"] if col in df.columns]
    if not stop_candidates:
        raise ValueError("Could not infer mutation columns. Provide --first_col and --last_col.")
    stop = min([list(df.columns).index(col) for col in stop_candidates])
    inferred = df.columns[:stop]
    if len(inferred) == 0:
        raise ValueError("Inferred zero mutation columns. Provide --first_col and --last_col.")
    print(f"Inferred OHE columns for {dataset}: first={inferred[0]} last={inferred[-1]} count={len(inferred)}")
    return inferred

def get_embedding_paths(dataset, base_path, model_names=None):
    embedding_root = os.path.join(base_path, "data", dataset, "embeddings")
    if model_names is not None:
        return {model_name: os.path.join(embedding_root, model_name) for model_name in model_names}
    if not os.path.isdir(embedding_root):
        raise FileNotFoundError(f"Embedding root does not exist: {embedding_root}")
    return {
        model_name: os.path.join(embedding_root, model_name)
        for model_name in sorted(os.listdir(embedding_root))
        if os.path.isdir(os.path.join(embedding_root, model_name))
    }

def as_classifier_labels(values, threshold=None):
    values = np.asarray(values)
    unique = np.unique(values[~pd.isna(values)])
    if len(unique) <= 2 and set(unique.tolist()).issubset({0, 1, 0.0, 1.0}):
        return values.astype(int)
    if threshold is None:
        threshold = float(np.nanmean(values))
    return (values > threshold).astype(int)

def precision_at_k(scores, y_true, k=100):
    scores = np.asarray(scores).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    if len(y_true) == 0:
        return np.nan
    k = min(int(k), len(y_true))
    if k <= 0:
        return np.nan
    pred_top = np.argsort(-scores)[:k]
    unique = np.unique(y_true[~pd.isna(y_true)])
    if len(unique) <= 2 and set(unique.tolist()).issubset({0, 1, 0.0, 1.0}):
        return float(np.mean(y_true[pred_top] == 1))
    true_top = np.argsort(-y_true)[:k]
    return float(len(np.intersect1d(pred_top, true_top)) / k)

def evaluate_regression(preds, y_true, precision_k=100):
    cor = spearmanr(preds, y_true)
    return {
        "correlation": cor.correlation,
        "p_value": cor.pvalue,
        "precision_at_k": precision_at_k(preds, y_true, k=precision_k),
    }

def evaluate_classification(proba_or_score, y_pred, y_true, precision_k=100):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    proba_or_score = np.asarray(proba_or_score)
    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, proba_or_score)
    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "precision_at_k": precision_at_k(proba_or_score, y_true, k=precision_k),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def fit_predict_mlp(x_train, y_train, x_test, hidden_layers, params, classifier=False):
    if classifier:
        unique = np.unique(y_train.astype(int))
        if len(unique) < 2:
            preds = np.full(x_test.shape[0], unique[0] if len(unique) else 0, dtype=int)
            scores = preds.astype(float)
            return scores, preds
        model = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers), **params)
        model.fit(x_train, y_train.astype(int))
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)[:, 1]
            preds = (scores > 0.5).astype(int)
        else:
            preds = model.predict(x_test)
            scores = preds
        return scores, preds
    model = MLPRegressor(hidden_layer_sizes=tuple(hidden_layers), **params)
    model.fit(x_train, y_train.astype(float))
    preds = model.predict(x_test)
    return preds, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="nmt", help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default=None, help='Explicit dataset CSV path. Defaults to <base_path>/data/<dataset>/<dataset>.csv for non-legacy datasets.')
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
    parser.add_argument('--first_col', type=str, default=None, help='First mutation column for OHE. Required for datasets not in the legacy defaults.')
    parser.add_argument('--last_col', type=str, default=None, help='Last mutation column for OHE. Required for datasets not in the legacy defaults.')
    parser.add_argument('--classifier', action='store_true', default=False, help='Run classification instead of regression.')
    parser.add_argument('--regression', action='store_true', default=False, help='Explicit no-op flag; regression is the default.')
    parser.add_argument('--classification_threshold', type=float, default=None, help='Threshold for converting continuous labels to binary labels. Defaults to label mean.')
    parser.add_argument('--precision_k', type=int, default=100, help='K for precision@K metrics.')
    parser.add_argument('--mean_embeddings', action='store_true', default=False,
                        help='If set, use mean embedding vectors (i.e., take mean on axis=1, not flatten)')
    # Add support for external labels (column from df, like in train_classifiers_over_embeddings.py)
    parser.add_argument('--external_labels_column', type=str, default=None,
                        help="If set, use this column from the dataframe as labels for all regression instead of the default activity/fitness column or embedding labels.")
    args = parser.parse_args()
    if args.precision_k <= 0:
        raise ValueError("--precision_k must be positive")

    base_path = args.base_path
    dataset = args.dataset_name
    is_classifier = args.classifier
    dataset_path = get_dataset_path(dataset, base_path, args.dataset_path)
    df = pd.read_csv(dataset_path)
    relevant_columns = get_relevant_columns_from_args(dataset, df, args.first_col, args.last_col)
    one_hot = get_one_hot_encoding(df, relevant_columns)
    assert one_hot.shape[1] == sum([len(pd.unique(df[C])) for C in relevant_columns])

    # Add sanity print if using external labels
    if args.external_labels_column is not None:
        label_values = df[args.external_labels_column].values
        print(f"Using external labels column: {args.external_labels_column}")
    else:
        label_values = get_label_column(dataset, df)
    if is_classifier:
        label_values = as_classifier_labels(label_values, args.classification_threshold)
    labels = torch.tensor(label_values).float()
    print("Sanity check: a few labels:", labels[:8].tolist())
    original_labels = labels.clone()
    embedding_paths = get_embedding_paths(dataset, base_path, args.model_names)

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
        indices_np = indices_emb.detach().cpu().numpy().astype(int)

        # Apply mean or flatten, depending on flag
        if args.mean_embeddings:
            emb_proc = embeddings.mean(axis=1)
        else:
            emb_proc = embeddings.reshape(embeddings.shape[0], -1)

        # Normalize embeddings
        normalized_embeddings = emb_proc - emb_proc.mean(dim=0, keepdim=True)
        normalized_embeddings = normalized_embeddings / emb_proc.std(dim=0, keepdim=True)
        if is_classifier:
            labels_emb = torch.tensor(as_classifier_labels(labels_emb.detach().cpu().numpy(), args.classification_threshold))
        labels_all[model_name] = labels_emb
        indices_all[model_name] = indices_emb
        embeddings_all[model_name] = normalized_embeddings

        if args.external_labels_column is not None or is_classifier:
            _ext_values = df.iloc[indices_np][args.external_labels_column].values if args.external_labels_column is not None else get_label_column(dataset, df.iloc[indices_np])
            if is_classifier:
                _ext_values = as_classifier_labels(_ext_values, args.classification_threshold)
            _ext_labels = torch.tensor(_ext_values).float()
            external_labels_all[model_name] = _ext_labels
            print(f"Sanity check on mapped labels for {model_name}:", _ext_labels[:8].tolist())

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
        task_name = "classification" if is_classifier else "regression"
        output_csv = f"data/{dataset}/{task_name}_result_by_training_samples.csv"
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
                ohe_scores, ohe_preds = fit_predict_mlp(
                    one_hot.numpy()[train_indices],
                    curr_labels.numpy()[train_indices],
                    one_hot.numpy()[test_indices],
                    args.ohe_hl,
                    ohe_mlp_params,
                    classifier=is_classifier,
                )
                ohe_true = curr_labels.numpy()[test_indices]
                if is_classifier:
                    for metric, value in evaluate_classification(ohe_scores, ohe_preds, ohe_true, precision_k=args.precision_k).items():
                        result_dict[f"{prefix}{metric}_ohe"] = value
                else:
                    ohe_metrics = evaluate_regression(ohe_preds, ohe_true, precision_k=args.precision_k)
                    result_dict[f"{prefix}cor_ohe"] = ohe_metrics["correlation"]
                    result_dict[f"{prefix}precision_at_k_ohe"] = ohe_metrics["precision_at_k"]

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

                    llm_scores, llm_preds = fit_predict_mlp(
                        normalized_embeddings.numpy()[emb_train_indices],
                        curr_llm_labels.numpy()[emb_train_indices],
                        normalized_embeddings.numpy()[emb_test_indices],
                        args.llm_hl,
                        llm_mlp_params,
                        classifier=is_classifier,
                    )
                    test_llm_labels = curr_llm_labels.numpy()[emb_test_indices]

                    # Record whether mean or flat
                    result_suffix = "_mean" if args.mean_embeddings else "_flat"
                    result_key = f"{model_name}{result_suffix}"
                    if is_classifier:
                        for metric, value in evaluate_classification(llm_scores, llm_preds, test_llm_labels, precision_k=args.precision_k).items():
                            result_dict[f"{prefix}{metric}_{result_key}"] = value
                    else:
                        llm_metrics = evaluate_regression(llm_preds, test_llm_labels, precision_k=args.precision_k)
                        result_dict[f"{prefix}cor_{result_key}"] = llm_metrics["correlation"]
                        result_dict[f"{prefix}precision_at_k_{result_key}"] = llm_metrics["precision_at_k"]
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
