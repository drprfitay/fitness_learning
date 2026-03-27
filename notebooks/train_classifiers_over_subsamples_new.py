import argparse
import pandas as pd
import numpy as np
import os
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def evaluate_classifier(score, predicted_label, gt_label, label_true=0, label_false=1):
    tp = sum((predicted_label == label_true) & (gt_label == label_true))
    tn = sum((predicted_label == label_false) & (gt_label == label_false))
    fp = sum((predicted_label == label_true) & (gt_label == label_false))
    fn = sum((predicted_label == label_false) & (gt_label == label_true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.
    accuracy = np.sum(predicted_label == gt_label) / len(predicted_label)
    roc = roc_auc_score(gt_label, score)
    ordered_scores = np.argsort(score)[0:100]
    top_100_pct = sum(gt_label[ordered_scores] == label_true) / 100

    evaluation = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "roc": roc,
        "top_100_pct": top_100_pct
    }
    return evaluation

def get_one_hot_encoding(sdf, first_col, last_col):
    si = np.where(sdf.columns == first_col)[0][0]
    ei = np.where(sdf.columns == last_col)[0][0]
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[sdf.columns[si:(ei+1)]]).to_numpy()).to(torch.int64)
    return one_hot_encoding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='gfp', type=str)
    parser.add_argument('--base_path', type=str, default="/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/")
    parser.add_argument('--embedding_dir', type=str, default=None, help="Directory with LLM embeddings")
    parser.add_argument('--embedding_model', type=str, default="esm_8m", choices=["esm_650m", "esm_35m", "esm_8m"])
    parser.add_argument('--n_samples', nargs='+', type=int, default=[10, 25, 50, 75, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000])
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--ohe_hl', nargs='+', type=int, default=[200])   # OHE hidden layer sizes
    parser.add_argument('--llm_hl', nargs='+', type=int, default=[64])    # LLM hidden layer sizes
    parser.add_argument('--output_csv', default=None, type=str)
    parser.add_argument('--shuffle_labels', action='store_true', help="Also perform label-shuffled controls")

    args = parser.parse_args()

    base_path = args.base_path
    sample_sizes = args.n_samples

    # Paths
    data_path = f"{base_path}/data/gfp/gfp_dataset_10mut.csv"
    if args.embedding_dir is not None:
        classifier_embeddings_path = args.embedding_dir
    else:
        classifier_embeddings_path = f"{base_path}/data/gfp/embeddings/{args.embedding_model}"

    # Load and aggregate all num_muts in [1, 10]
    df = pd.read_csv(data_path)
    one_hot = get_one_hot_encoding(df, "L42", "V224").numpy()
    si = np.where(df.columns == "L42")[0][0]
    ei = np.where(df.columns == "V224")[0][0]
    assert one_hot.shape[1] == sum([len(pd.unique(df[C])) for C in df.columns[si:(ei+1)]])

    ohe_labels_full = df["inactive"].astype(int).to_numpy()

    # Aggregate indices, OHE, LLM embeddings and labels for all n_train in 1-10:
    agg_embedding_arr = []
    agg_ohe_arr = []
    agg_labels_arr = []
    agg_indices_arr = []  # indices in df

    for n_train in range(1, 11):
        train_indices = np.where((df["num_muts"] == n_train).to_numpy())[0]
        sub_ohe_labels = ohe_labels_full[train_indices]
        sub_one_hot = one_hot[train_indices]

        labels = torch.load(os.path.join(classifier_embeddings_path, f"y_values_of_nmut_{n_train}.pt"))
        indices = torch.load(os.path.join(classifier_embeddings_path, f"indices_of_nmut_{n_train}.pt"))
        embeddings = torch.load(os.path.join(classifier_embeddings_path, f"embeddings_of_nmut_{n_train}.pt"))

        # Sanity checks
        assert sum(labels == sub_ohe_labels) == len(sub_ohe_labels)
        assert sum(indices == train_indices) == len(train_indices)

        agg_embedding_arr.append(embeddings.numpy())
        agg_ohe_arr.append(sub_one_hot)
        agg_labels_arr.append(sub_ohe_labels)
        agg_indices_arr.append(train_indices)

    embeddings_all = np.concatenate(agg_embedding_arr, axis=0)
    ohe_all = np.concatenate(agg_ohe_arr, axis=0)
    labels_all = np.concatenate(agg_labels_arr, axis=0)
    indices_all = np.concatenate(agg_indices_arr, axis=0)  # for reference to original df

    flat_embeddings = embeddings_all.reshape(embeddings_all.shape[0], -1)
    normalized_embeddings = flat_embeddings - flat_embeddings.mean(axis=0, keepdims=True)
    normalized_embeddings = normalized_embeddings / flat_embeddings.std(axis=0, keepdims=True)

    mlp_base_parameters = dict(
        activation='relu',
        solver='lbfgs',
        batch_size=128,
        alpha=1,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=200,
        random_state=42,
        verbose=True
    )

    # Shuffling modes
    shuffle_modes = [True, False] if args.shuffle_labels else [False]

    for sample_size in sample_sizes:
        across_iterations_llm = []
        across_iterations_ohe = []

        for iter_num in range(args.iters):
            print(f"Iteration {iter_num}, sample_size {sample_size}")

            # Sample train indices from aggregated pool (without replacement)
            n_total = embeddings_all.shape[0]
            n_train = min(sample_size, n_total)
            train_indices_agg = np.random.choice(np.arange(n_total), size=n_train, replace=False)
            test_indices_agg = np.setdiff1d(np.arange(n_total), train_indices_agg)

            # Data splits
            train_ohe = ohe_all[train_indices_agg]
            test_ohe = ohe_all[test_indices_agg]
            train_llm = normalized_embeddings[train_indices_agg]
            test_llm = normalized_embeddings[test_indices_agg]
            

            result_dict_base = {
                'sample_size': sample_size,
                'iter': iter_num
            }

            for shuffle in shuffle_modes:
                mode = "shuffled" if shuffle else "regular"

                labels_to_destory = labels_all.copy()

                if shuffle:
                    curr_labels = np.random.permutation(labels_to_destory)
                else:
                    curr_labels = labels_to_destory

                # OHE

                train_labels = curr_labels[train_indices_agg]
                test_labels = curr_labels[test_indices_agg]
                
                
                mlp_ohe = MLPClassifier(hidden_layer_sizes=tuple(args.ohe_hl), **mlp_base_parameters)
                mlp_ohe.fit(train_ohe, train_labels)

                predictions_proba = mlp_ohe.predict_proba(test_ohe)
                pred_scores = predictions_proba[:, 1]
                pred_labels = (pred_scores > 0.5).astype(int)

                eval_ohe = evaluate_classifier(pred_scores, pred_labels, test_labels)
                eval_ohe['shuffle'] = mode
                eval_ohe.update(result_dict_base)
                across_iterations_ohe.append(eval_ohe)

                # LLM
                mlp_llm = MLPClassifier(hidden_layer_sizes=tuple(args.llm_hl), **mlp_base_parameters)
                mlp_llm.fit(train_llm, train_labels)

                predictions_proba = mlp_llm.predict_proba(test_llm)
                pred_scores = predictions_proba[:, 1]
                pred_labels = (pred_scores > 0.5).astype(int)

                eval_llm = evaluate_classifier(pred_scores, pred_labels, test_labels)
                eval_llm['shuffle'] = mode
                eval_llm.update(result_dict_base)
                across_iterations_llm.append(eval_llm)

                print(f"OHE evaluation%s: {eval_ohe}" % ( " (shuffled)" if shuffle else ""))
                print(f"LLM evaluation%s: {eval_llm}" % ( " (shuffled)" if shuffle else ""))

        # Save results for this sample_size
        result_base = args.output_csv if args.output_csv is not None else "data/gfp/subsamples/"
        os.makedirs(result_base, exist_ok=True)
        suffix = f"_{sample_size}.csv"

        pd.DataFrame(across_iterations_llm).to_csv(os.path.join(result_base, f'llm_results{suffix}'), index=False)
        pd.DataFrame(across_iterations_ohe).to_csv(os.path.join(result_base, f'ohe_results{suffix}'), index=False)
        print(f"\nSaved results to {result_base} (sample_size={sample_size})\n")

if __name__ == "__main__":
    main()
