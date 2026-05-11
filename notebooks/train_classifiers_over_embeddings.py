import argparse
import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, RidgeClassifier

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.decomposition import PCA

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from utils_for_analysis import *

# Use xgboost from scikit-learn interface (since 1.3.0: sklearn.ensemble)
try:
    from sklearn.ensemble import HistGradientBoostingClassifier as SkXGBClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor as SkXGBRegressor
    sklearn_xgb_available = True
except ImportError:
    sklearn_xgb_available = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', nargs='+', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default="gfp", help='Dataset name')
    parser.add_argument('--delta_embeddings', action='store_true', default=False, help='Flag to overwrite existing results file')
    parser.add_argument('--mean_embeddings', action='store_true', default=False, help='Flag to use mean embeddings')
    parser.add_argument('--regression', action='store_true', default=False, help='Flag to use regression instead of classification')
    parser.add_argument('--external_labels_column', type=str, default=None,
                        help='If set, use the specified column from the original dataset as labels')
    parser.add_argument('--hl', type=int, nargs='+', default=[100], 
                        help='List of hidden layer sizes for the MLP (e.g. --hidden_layers 100 50)')
    parser.add_argument('--op', type=str, default='adam', choices=['adam', 'lbfgs'],
                        help='Optimizer to use in training the classifier (adam or sgd)')
    parser.add_argument('--niters', type=int, default=50,
                        help='Number of iterations (epochs) for training the classifier')
    parser.add_argument('--n_start', type=int, default=2,
                        help='Start value for the number of training mutations (inclusive)')
    parser.add_argument('--n_end', type=int, default=3,
                        help='End value for the number of training mutations (inclusive)')
    parser.add_argument("--pca", type=int, default=None, help="Number of PCA components to use")
    parser.add_argument(
        '--classifier_model', 
        type=str, 
        default='mlp', 
        choices=['mlp', 'ridgeregression', 'logisticregression', 'xgboost', 'lasso'],
        help="Choose classifier/regressor: mlp, ridgeregression, logisticregression, xgboost, lasso"
    )

    args = parser.parse_args()

    base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"
    embedding_paths = {
        "esm_650m" : "%s/data/%s/embeddings/esm_650m" % (base_path, args.dataset_name),
        "esm_8m" : "%s/data/%s/embeddings/esm_8m" % (base_path, args.dataset_name),
        "esm_35m" : "%s/data/%s/embeddings/esm_35m" % (base_path, args.dataset_name),
        "esm_150m" : "%s/data/%s/embeddings/esm_150m" % (base_path, args.dataset_name),
        "esm_3b" : "%s/data/%s/embeddings/esm_3b" % (base_path, args.dataset_name),
        "progen2-small" : "%s/data/%s/embeddings/progen2-small" % (base_path, args.dataset_name),
        "progen2-medium" : "%s/data/%s/embeddings/progen2-medium" % (base_path, args.dataset_name),
        "ankh3-large" : "%s/data/%s/embeddings/ankh3-large" % (base_path, args.dataset_name),
        "prot_bert" : "%s/data/%s/embeddings/prot_bert" % (base_path, args.dataset_name)
    }

    external_labels = None

    if args.external_labels_column is not None:
        csv_path = "%s/data/%s/%s.csv" % (base_path, args.dataset_name, args.dataset_name)

        df = pd.read_csv(csv_path)
        external_labels = df[args.external_labels_column].values

        if not args.regression:
            external_labels = (external_labels > np.mean(external_labels)).astype(int)
            
        print("Using external_lables:")
        print(external_labels)

    for model_name in args.model_name:
        print(f"Training classifiers over {model_name} embeddings")

        if model_name not in embedding_paths:
            raise ValueError(f"Model name '{model_name}' not found in available embedding paths: {list(embedding_paths.keys())}")

        classifier_embeddings_path = embedding_paths[model_name]
        print(f"Classifier embeddings path: {classifier_embeddings_path}")

        if args.delta_embeddings:
            wt_embedding = torch.load(os.path.join(classifier_embeddings_path, "embeddings_of_nmut_0.pt"))


        for n_train in range(args.n_start, args.n_end):
            train_data = {}
            n_train_data = n_train
            n_test_data = list(range(n_train + 1, 11))

            for i in range(1, n_train_data + 1):
                labels = torch.load(os.path.join(classifier_embeddings_path, "y_values_of_nmut_%d.pt" % i))
                indices = torch.load(os.path.join(classifier_embeddings_path, "indices_of_nmut_%d.pt" % i))
                embeddings = torch.load(os.path.join(classifier_embeddings_path, "embeddings_of_nmut_%d.pt" % i))

                if external_labels is not None:
                    labels = torch.from_numpy(external_labels[indices])
                
                if args.delta_embeddings:
                    print(f"Subtracting WT embedding from {model_name} embeddings")
                    embeddings = embeddings - wt_embedding

                train_data["nmuts_%d" % i] = {
                    "labels": labels,
                    "indices": indices,
                    "embeddings": embeddings
                }
                
            indices_all = []
            labels_all = []
            embeddings_all = []

            for k, v in train_data.items():
                indices_all.append(v["indices"])
                labels_all.append(v["labels"])
                embeddings_all.append(v["embeddings"])

            indices_all = torch.cat(indices_all)
            embeddings_all = torch.cat(embeddings_all)
            labels_all = torch.cat(labels_all)

            mlp_base_parameters = {
                "activation" : 'relu',           
                "solver" : args.op, 
                "batch_size": 128,   
                "alpha" : 1,                
                "learning_rate_init" : 1e-3,    
                "max_iter" :args.niters,
                "random_state" : 4321,                
                "n_iter_no_change" : 10,         
                "verbose": False
            }

            if args.mean_embeddings:
                flat_embeddings = embeddings_all.mean(axis=1)
            else:
                flat_embeddings = embeddings_all.reshape(embeddings_all.shape[0], -1)
            normalized_embeddings = flat_embeddings - flat_embeddings.mean(dim=0, keepdim=True)
            train_std = flat_embeddings.std(dim=0, keepdim=True)
            normalized_embeddings = normalized_embeddings / train_std

            if args.pca is not None:
                train_pca = PCA(n_components=args.pca)
                print("########################")
                print(normalized_embeddings.shape)
                print("########################")
                train_pca.fit(normalized_embeddings.numpy())
                normalized_embeddings = ((normalized_embeddings - train_pca.mean_) @ train_pca.components_.T)

            print("Fitting model (%s) over %dx%d embeddings to %d labels [%d mutations]" % 
                        (args.classifier_model,
                         normalized_embeddings.shape[0], 
                         normalized_embeddings.shape[1], 
                         len((labels_all)),
                         n_train_data))

            if isinstance(labels_all, torch.Tensor):                
                print(labels_all.shape)
                print(type(labels_all))
            else:
                print(labels_all.shape)
                print(type(labels_all))

            # Select classifier/regressor according to flag
            model = None
            if args.classifier_model == 'mlp':
                if args.regression:
                    model = MLPRegressor(hidden_layer_sizes=args.hl, **mlp_base_parameters)
                    model.fit(normalized_embeddings.numpy(), labels_all.numpy().astype(float))
                else:
                    model = MLPClassifier(hidden_layer_sizes=args.hl, **mlp_base_parameters)
                    model.fit(normalized_embeddings.numpy(), labels_all)
            elif args.classifier_model == 'ridgeregression':
                if args.regression:
                    model = Ridge()
                    model.fit(normalized_embeddings.numpy(), labels_all.numpy().astype(float))
                else:
                    model = RidgeClassifier(alpha=1.0, solver="auto", max_iter=args.niters)
                    model.fit(normalized_embeddings.numpy(), labels_all.numpy())
            elif args.classifier_model == 'lasso':
                if not args.regression:
                    raise ValueError("Lasso is only supported for regression tasks. Use --regression.")
                model = Lasso()
                model.fit(normalized_embeddings.numpy(), labels_all.numpy().astype(float))
            elif args.classifier_model == 'logisticregression':
                if args.regression:
                    raise ValueError("LogisticRegression is only supported for classification tasks. Do not use --regression.")
                model = LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=args.niters,
                    n_jobs=-1
                )
                model.fit(normalized_embeddings.numpy(), labels_all.numpy())
            elif args.classifier_model == 'xgboost':
                if not sklearn_xgb_available:
                    raise ImportError("scikit-learn's HistGradientBoosting (recommended xgboost-alternative) is not available. Please ensure scikit-learn >=0.21 is installed.")
                if args.regression:
                    model = GradientBoostingRegressor(n_estimators=args.niters)
                    model.fit(normalized_embeddings.numpy(), labels_all.numpy().astype(float))
                else:
                    model = GradientBoostingClassifier(n_estimators=args.niters)
                    model.fit(normalized_embeddings.numpy(), labels_all.numpy())
            else:
                raise ValueError("Unknown classifier_model selected: %s" % args.classifier_model)

            test_data = []
            for i in n_test_data:
                labels = torch.load(os.path.join(classifier_embeddings_path, "y_values_of_nmut_%d.pt" % i))
                indices = torch.load(os.path.join(classifier_embeddings_path, "indices_of_nmut_%d.pt" % i))
                embeddings = torch.load(os.path.join(classifier_embeddings_path, "embeddings_of_nmut_%d.pt" % i))

                if args.mean_embeddings:
                    flat_embeddings = embeddings.mean(axis=1)
                else:
                    flat_embeddings = embeddings.reshape(embeddings.shape[0], -1)
                    
                normalized_embeddings = flat_embeddings - flat_embeddings.mean(dim=0, keepdim=True)
                normalized_embeddings = normalized_embeddings / train_std

                if args.pca is not None:
                    normalized_embeddings = (normalized_embeddings - train_pca.mean_) @ train_pca.components_.T

                # Predict and evaluate
                if args.regression:
                    preds = model.predict(normalized_embeddings.numpy())
                    preds_np = np.asarray(preds)
                    nan_indices = np.isnan(preds_np)
                    inf_indices = np.isinf(preds_np)
                    print("Indices of NaN in preds:", np.where(nan_indices)[0])
                    print("Indices of Inf in preds:", np.where(inf_indices)[0])
                    print(labels)
                    print(labels.numpy().mean())
                    print(labels.numpy().std())
                    print(preds_np)
                    print(preds_np.mean())
                    print(preds_np.std())
                    print(preds_np)
                    preds_np = np.asarray(labels)
                    nan_indices = np.isnan(preds_np)
                    inf_indices = np.isinf(preds_np)
                    print("Indices of NaN in preds:", np.where(nan_indices)[0])
                    print("Indices of Inf in preds:", np.where(inf_indices)[0])
                    print("########################")
                    correlation = spearmanr(preds, labels.numpy())
                    evaluation = {
                        "correlation": correlation[0],
                        "p_value": correlation[1],
                    }
                    print(correlation)
                    print("########################")
                else:
                    # For classifiers, always use probabilities and 0/1 threshold at 0.5, if possible
                    try:
                        predictions_proba = model.predict_proba(normalized_embeddings.numpy())
                        predictions = (predictions_proba[:, 1] > 0.5).astype(int)
                        evaluation = evaluate_classifier(predictions_proba[:, 1], predictions, labels.numpy())
                    except AttributeError:
                        # Some classifiers might not have predict_proba, e.g., some linear models or HistGradientBoostingClassifier returns both predict_proba (for binary) or decision_function (for multiclass)
                        try:
                            predictions_proba = model.predict_proba(normalized_embeddings.numpy())
                            predictions = (predictions_proba[:, 1] > 0.5).astype(int)
                            evaluation = evaluate_classifier(predictions_proba[:, 1], predictions, labels.numpy())
                        except Exception:
                            predictions = model.predict(normalized_embeddings.numpy())
                            evaluation = evaluate_classifier(predictions, predictions, labels.numpy())

                print("Evaluation for %d mutations: %s" % (i, evaluation))

                test_data.append(evaluation)

                evaluation["test_mutations"] = i
                evaluation["train_mutations"] = n_train_data
                evaluation["classifier"] = classifier_embeddings_path
                evaluation["model_name"] = model_name
                evaluation["clf_type"] = args.classifier_model

            folder_name = "delta_grid_searched" if args.delta_embeddings else "new_grid_searched"
            folder_name = folder_name + "_mean" if args.mean_embeddings else folder_name + "_flat"

            if args.regression:
                save_dir = f"data/{args.dataset_name}/{args.dataset_name}_regression_results_32/{folder_name}/{model_name}"
            else:
                save_dir = f"data/{args.dataset_name}_classification_results/{folder_name}/{model_name}"
            if args.pca is not None:
                save_dir = save_dir + f"_pca_{args.pca}"
            print(f"Saving results to {save_dir}")

            os.makedirs(save_dir, exist_ok=True)

            test_data = pd.DataFrame(test_data)
            print(test_data)

            test_data.to_csv(os.path.join(save_dir, f"{model_name}_evaluation_train_on_{n_train_data}_{args.classifier_model}.csv"), index=False)

if __name__ == '__main__':
    main()
