import argparse
import pandas as pd
import numpy as np
import os
import torch

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import spearmanr
import itertools
from utils_for_analysis import * 


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
        cols = get_relevant_columns_gfp_protgym(df, positions[dataset][0], positions[dataset][1])
        return cols
    else:
        raise ValueError(f"Unknown dataset for OHE column range: {dataset}")

def get_num_muts_column_name(dataset):
    return {
        "pard3": "num_muts",
        "lov": "num_muts",
        "gfp": "num_muts",
        "pte": "num_muts",
        "gcn4": "num_muts",
        "nmt": "num_muts"
    }[dataset]

# --- OHE Feature Creation ---
def get_one_hot_encoding(sdf, relevant_columns):
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[relevant_columns]).to_numpy()).to(torch.float32)
    return one_hot_encoding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="gfp", help='Dataset name (gfp, pte, nmt, etc)')
    parser.add_argument('--regression', action='store_true', default=False, help='Flag to use regression instead of classification')
    parser.add_argument('--external_labels_column', type=str, default=None,
                        help='If set, use the specified column from the original dataset as labels')
    parser.add_argument('--ohe_save_dir', type=str, default=None, help="Extra results subfolder name for OHE outputs")
    parser.add_argument('--only_train_n', type=int, default=None, help='If set, only do one value of n_train')
    
    # We allow the user to specify grid of hidden layers, alpha, solver, learning_rate, niters
    parser.add_argument('--hl', type=str, nargs='+', default=["100"], 
                        help="List of hidden layer sizes for the MLP; space separated. Accepts format: 100 or 100,50")
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5], help="List of alpha values to grid search over")
    parser.add_argument('--op', type=str, nargs='+', default=['adam'], choices=['adam', 'lbfgs'],
                        help='Optimizers to try in the grid search')
    parser.add_argument('--niters', type=int, nargs='+', default=[50],
                        help='List of #iterations (epochs) for training the classifier')
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-3], help="List of learning_rates to grid search over")
    parser.add_argument('--n_start', type=int, default=2,
                        help='Start value for the number of training mutations (inclusive)')
    parser.add_argument('--n_end', type=int, default=3,
                        help='End value for the number of training mutations (inclusive)')
    args = parser.parse_args()

    # Parse hidden layers
    hidden_layers_list = []
    for hl in args.hl:
        if ',' in hl:
            hidden_layers_list.append(tuple(int(i) for i in hl.split(',')))
        else:
            hidden_layers_list.append((int(hl),))

    dataset_path = DATASET_PATHS[args.dataset_name]
    df = pd.read_csv(dataset_path)

    # get one-hot encoding for relevant columns
    relevant_columns = get_relevant_columns(args.dataset_name, df)
    ohe = get_one_hot_encoding(df, relevant_columns)
    assert ohe.shape[1] == sum([len(pd.unique(df[C])) for C in relevant_columns])

    # Figure out the label column
    if args.external_labels_column is not None:
        labels_all = df[args.external_labels_column].values
    else:
        if args.dataset_name == "gfp":
            labels_all = df["inactive"].astype(int).values
        else:
            raise ValueError("Specify --external_labels_column for non-gfp datasets")
    is_regression = args.regression
    if (not is_regression) and args.external_labels_column:
        labels_all = (labels_all > np.mean(labels_all)).astype(int)

    num_muts_col = get_num_muts_column_name(args.dataset_name)
    df[num_muts_col] = df[num_muts_col].astype(int)

    # We'll aggregate grid searched parameter sets + out csv path + results dataframe here
    param_aggregation = []

    # Prepare grid of params
    param_grid = list(itertools.product(
        hidden_layers_list,
        args.alpha,
        args.op,
        args.niters,
        args.lr
    ))

    for n_train in range(args.n_start, args.n_end):
        n_train_data = n_train
        n_test_data = list(range(n_train + 1, 11))
        train_indices = (df[num_muts_col] <= n_train) & (df[num_muts_col] > 0)
        X_train = ohe[train_indices]
        y_train = labels_all[train_indices]

        for grid_idx, (hidden_layers, alpha, solver, niters, lr) in enumerate(param_grid):
            mlp_base_parameters = {
                "activation": 'relu',
                "solver": solver, 
                "batch_size": 128,
                "alpha": alpha,
                "learning_rate_init": lr,
                "max_iter": niters,
                "random_state": 4321,
                "n_iter_no_change": 10,
                "verbose": False
            }

            print(mlp_base_parameters)

            print(f"\n--- Grid {grid_idx}: Params: hidden_layers={hidden_layers}, "
                  f"alpha={alpha}, solver={solver}, niters={niters}, lr={lr} ---")
            print(f"Fitting {'regressor' if is_regression else 'classifier'} MLP over {X_train.shape[0]} x {X_train.shape[1]} OHE features, train size={len(y_train)} [train n_muts <= {n_train}]")

            if is_regression:
                mlp_ohe = MLPRegressor(hidden_layer_sizes=(200, 200, 10), **mlp_base_parameters)
                mlp_ohe.fit(X_train.numpy(), y_train.astype(float))
            else:
                mlp_ohe = MLPClassifier(hidden_layer_sizes=(200, 200, 10), **mlp_base_parameters)
                mlp_ohe.fit(X_train.numpy(), y_train)

            test_data = []

            for i in n_test_data:
                test_indices = df[num_muts_col] == i
                X_test = ohe[test_indices]
                y_test = labels_all[test_indices]
                if len(y_test) == 0:
                    print(f"Skipping test n_muts={i}, no samples.")
                    continue

                if is_regression:
                    predictions = mlp_ohe.predict(X_test.numpy())
                    correlation = spearmanr(predictions, y_test)
                    evaluation = {
                        "correlation": correlation[0],
                        "p_value": correlation[1],
                    }
                else:
                    predictions_proba = mlp_ohe.predict_proba(X_test.numpy())
                    predictions = (predictions_proba[:, 1] > 0.5).astype(int)
                    evaluation = evaluate_classifier(predictions_proba[:, 1], predictions, y_test)

                print(f"Evaluation for {i} mutations: {evaluation}")
                evaluation["test_mutations"] = i
                evaluation["train_mutations"] = n_train_data
                evaluation["classifier"] = "ohe"
                evaluation["dataset"] = args.dataset_name
                # Also record grid search parameters in the evaluation for tracking
                evaluation["grid_hidden_layers"] = str(hidden_layers)
                evaluation["grid_alpha"] = alpha
                evaluation["grid_solver"] = solver
                evaluation["grid_niters"] = niters
                evaluation["grid_lr"] = lr

                test_data.append(evaluation)

            test_data_df = pd.DataFrame(test_data)
            # Folder naming: consistent with embedding script
            model_str = f"grid_searched"
            results_type = "regression_results" if is_regression else "classification_results"
            ohe_folder = args.ohe_save_dir if args.ohe_save_dir else model_str
            folder_name = ohe_folder
            grid_hl_str = "_".join(str(x) for x in hidden_layers)
            grid_alpha_str = f"alpha_{alpha}"
            grid_solver_str = f"solver_{solver}"
            grid_lr_str = ("lr_{:.1e}".format(lr) if lr < 0.001 else f"lr_{lr}")
            grid_iter_str = f"iter_{niters}"

            model_name = f"ohe_{grid_hl_str}_{grid_alpha_str}_{grid_solver_str}_{grid_lr_str}_{grid_iter_str}"
            save_dir = os.path.join(
                "data",
                args.dataset_name,
                f"{args.dataset_name}_regression_results",
                folder_name,
                model_name
            )
            os.makedirs(save_dir, exist_ok=True)
            out_csv = os.path.join(save_dir, f"ohe_evaluation_train_on_{n_train_data}.csv")
            test_data_df.to_csv(out_csv, index=False)
            print(f"Saved evaluation: {out_csv}")
            print(test_data_df)

            # Aggregate the grid parameters and metadata into dictionary
            param_aggregation.append({
                "params": {
                    "hidden_layers": hidden_layers,
                    "alpha": alpha,
                    "solver": solver,
                    "niters": niters,
                    "lr": lr
                },
                "train_n": n_train_data,
                "eval_csv": out_csv,
                "results_dataframe": test_data_df
            })
    # Optionally, after all, print or save the parameter results
    print(f"\nTotal parameter grid runs: {len(param_aggregation)}")


    # Save param_aggregation to a file for later inspection
    param_agg_save_path = os.path.join(
        "data",
        args.dataset_name,
        f"{args.dataset_name}_regression_results",
        folder_name,
        f"param_aggregation_train_on_{n_train_data}.pkl"
    )
    # Dump using pandas; DataFrames inside may need special handling
    import pickle
    with open(param_agg_save_path, "wb") as f:
        pickle.dump(param_aggregation, f)
    print(f"Saved param_aggregation to: {param_agg_save_path}")
    # You could save param_aggregation here if needed!

if __name__ == '__main__':
    main()