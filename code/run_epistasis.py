import pandas as pd
import numpy as np
import os
import torch
import seaborn as sns
import random
import pickle
import matplotlib.gridspec as gridspec


from scipy.stats import pearsonr, spearmanr
from scipy.stats import rankdata
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from Bio import SeqIO

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr

from collections import defaultdict

warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
    "aamyl": {"path": "./data/aamyl/aamyl.csv", "activity_col": "activity"},
    "trpb": {"path": "./data/trpb/trpb.csv", "activity_col": "activity"},
    "nucb": {"path": "./data/nucb/nucb.csv", "activity_col": "activity"},
    "ired": {"path": "./data/ired/ired.csv", "activity_col": "activity"},
    "his2": {"path": "./data/his2/his2.csv", "activity_col": "activity"},
    "his5": {"path": "./data/his5/his5.csv", "activity_col": "activity"},
    "casp": {"path": "./data/casp/casp.csv", "activity_col": "activity"}
}


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score


# ============================================================
# Metrics and splitting
# ============================================================

def make_activity_labels(activity, percentile=0.50):
    """
    For continuous activity:
        positive = activity above the requested percentile.

    For binary 0/1 activity:
        preserve the original labels.
    """
    activity = np.asarray(activity, dtype=float)
    valid = np.isfinite(activity)
    unique_values = np.unique(activity[valid])

    labels = np.zeros(len(activity), dtype=int)

    if set(unique_values).issubset({0, 1}):
        threshold = 0.5
        labels[valid] = activity[valid].astype(int)
    else:
        threshold = np.quantile(activity[valid], percentile)
        labels[valid] = (
            activity[valid] > threshold
        ).astype(int)

    return labels, threshold, valid


def resolve_random_train_amount(amount, n_total):
    """
    Automatically interpret a random-training amount.

    0 < amount <= 1: fraction of the complete dataset
    amount > 1:      number of training sequences

    The value 1 means 100%, not one sequence.
    """
    amount = float(amount)

    if amount <= 0:
        raise ValueError("Training amount must be positive.")

    if amount <= 1:
        amount_type = "fraction"
        n_train = int(round(n_total * amount))
    else:
        if not amount.is_integer():
            raise ValueError(
                "Values above 1 must be integer sequence counts."
            )

        amount_type = "number"
        n_train = int(amount)

    # Random splitting requires at least one held-out sequence
    n_train = np.clip(n_train, 1, n_total - 1)

    return int(n_train), amount_type

def score_predictions(
    activity,
    labels,
    train_idx,
    test_idx,
    predictions
):
    """Calculate metrics from already-generated predictions."""
    if len(test_idx) > 1:
        r2 = r2_score(
            activity[test_idx],
            predictions
        )
    else:
        r2 = np.nan

    if (
        len(test_idx) > 1 and
        len(np.unique(activity[test_idx])) > 1 and
        len(np.unique(predictions)) > 1
    ):
        spearman = spearmanr(
            activity[test_idx],
            predictions
        ).statistic
    else:
        spearman = np.nan

    if len(np.unique(labels[test_idx])) == 2:
        auc = roc_auc_score(
            labels[test_idx],
            predictions
        )
    else:
        auc = np.nan

    return {
        "r2": r2,
        "spearman": spearman,
        "auc": auc,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "positive_rate_train": labels[train_idx].mean(),
        "positive_rate_test": labels[test_idx].mean()
    }


def fit_and_evaluate(
    X,
    activity,
    labels,
    train_idx,
    test_idx
):
    model = LinearRegression()
    model.fit(
        X[train_idx],
        activity[train_idx]
    )

    predictions = model.predict(
        X[test_idx]
    )

    metrics = score_predictions(
        activity=activity,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        predictions=predictions
    )

    return model, metrics


def collect_mutation_classification_errors(
    X,
    labels,
    mut_order,
    valid,
    source_indices,
    dataset_name,
    train_ks=range(1, 5),
    random_state=0
):
    """
    Fit one logistic regression for every <=K training pool and record
    the original row indices of its training classes and test errors.

    Each model is trained on every valid sequence with num_muts <= K.
    It is then evaluated separately on every exact test order > K.
    """
    rows = []

    for k in train_ks:
        train_idx = np.where(
            valid & (mut_order <= k)
        )[0]

        if len(train_idx) == 0:
            continue

        positive_train_idx = source_indices[
            train_idx[labels[train_idx] == 1]
        ].tolist()
        negative_train_idx = source_indices[
            train_idx[labels[train_idx] == 0]
        ].tolist()

        train_classes = np.unique(labels[train_idx])

        if len(train_classes) == 2:
            model = LogisticRegression(
                C=1.0,
                solver="saga",
                class_weight="balanced",
                max_iter=3000,
                random_state=random_state
            )
            model.fit(X[train_idx], labels[train_idx])
        else:
            model = None
            constant_prediction = int(train_classes[0])

        test_orders = np.sort(
            np.unique(mut_order[valid & (mut_order > k)])
        )

        for test_order in test_orders:
            test_idx = np.where(
                valid & (mut_order == test_order)
            )[0]

            if model is None:
                predictions = np.full(
                    len(test_idx),
                    constant_prediction,
                    dtype=int
                )
            else:
                predictions = model.predict(X[test_idx])

            false_positive_idx = source_indices[
                test_idx[
                    (predictions == 1) &
                    (labels[test_idx] == 0)
                ]
            ].tolist()

            false_negative_idx = source_indices[
                test_idx[
                    (predictions == 0) &
                    (labels[test_idx] == 1)
                ]
            ].tolist()

            rows.append({
                "dataset": dataset_name,
                "mutations_in_train": int(k),
                "positive_indices_train": positive_train_idx,
                "negative_indices_train": negative_train_idx,
                "false_positive_indices_test": false_positive_idx,
                "false_negative_indices_test": false_negative_idx,
                "test_order": int(test_order)
            })

    return pd.DataFrame(rows)


def run_linear_experiments(
    datasets,
    percentile=0.50,
    random_train_amounts=None,
    mutation_train_fractions=(1.0,),
    mutation_ks=None,
    classification_train_ks=range(1, 5),
    collect_classification_errors=True,
    n_repeats=10,
    max_mutations=10,
    random_seed=0
):
    """
    Regimes
    -------
    all_all:
        Train and evaluate on all variants.

    random:
        Train on a random fraction or number of variants and evaluate
        on the complementary held-out set.

    mutation:
        Train on variants with num_muts <= K and evaluate:

        1. Jointly on all variants with num_muts > K.
        2. Separately on each exact test mutation order > K.
    """
    if random_train_amounts is None:
        random_train_amounts = np.arange(
            0.05,
            1.00,
            0.05
        )

    rows = []
    classification_error_frames = []

    for dataset_i, dataset_name in enumerate(datasets):
        print(f"Working on {dataset_name}")

        rng = np.random.default_rng(
            random_seed + dataset_i
        )

        df = pd.read_csv(
            DATASET_PATHS[dataset_name]
        )

        df = df[
            df["num_muts"] <= max_mutations
        ].copy()
        df["_source_index"] = df.index
        df = df.reset_index(drop=True)
        source_indices = df["_source_index"].to_numpy()

        relevant_columns = get_relevant_columns(
            dataset_name,
            df
        )

        X = get_one_hot_encoding(
            df,
            relevant_columns
        )

        if isinstance(X, pd.DataFrame):
            all_sparse = all(
                isinstance(dtype, pd.SparseDtype)
                for dtype in X.dtypes
            )

            if all_sparse:
                X = (
                    X.sparse
                    .to_coo()
                    .tocsr()
                    .astype(np.float32)
                )
            else:
                X = sparse.csr_matrix(
                    X.to_numpy(
                        dtype=np.float32,
                        copy=False
                    )
                )
        elif sparse.issparse(X):
            X = X.tocsr().astype(np.float32)
        else:
            X = sparse.csr_matrix(
                X,
                dtype=np.float32
            )

        mut_order = df[
            "num_muts"
        ].to_numpy()

        if dataset_name == "gfp":
            activity = (
                1 - df["inactive"]
            ).to_numpy(dtype=float)

            # GFP already has binary experimental labels. An active GFP
            # sequence is 1 and an inactive GFP sequence is 0; no
            # percentile threshold is used.
            valid = np.isfinite(activity)
            labels = np.zeros(len(activity), dtype=int)
            labels[valid] = activity[valid].astype(int)
            threshold = 0.5
        else:
            activity_col = datasets_and_activity[
                dataset_name
            ]["activity_col"]

            activity = df[
                activity_col
            ].to_numpy(dtype=float)

            labels, threshold, valid = make_activity_labels(
                activity,
                percentile
            )

        all_idx = np.where(valid)[0]

        if collect_classification_errors:
            classification_error_frames.append(
                collect_mutation_classification_errors(
                    X=X,
                    labels=labels,
                    mut_order=mut_order,
                    valid=valid,
                    source_indices=source_indices,
                    dataset_name=dataset_name,
                    train_ks=classification_train_ks,
                    random_state=random_seed + dataset_i
                )
            )

        def append_result(
            metrics,
            regime,
            repeat,
            requested_train_amount,
            train_amount_type,
            training_pool_size,
            k=np.nan,
            test_scope="held_out_complement",
            test_order=np.nan
        ):
            rows.append({
                "dataset": dataset_name,
                "regime": regime,
                "repeat": repeat,
                "requested_train_amount": requested_train_amount,
                "train_amount_type": train_amount_type,
                "train_fraction": (
                    metrics["n_train"] /
                    training_pool_size
                ),
                "k": k,
                "test_scope": test_scope,
                "test_order": test_order,
                "percentile": percentile,
                "activity_threshold": threshold,
                **metrics
            })

        # ====================================================
        # Train all, test all
        # ====================================================

        _, metrics = fit_and_evaluate(
            X=X,
            activity=activity,
            labels=labels,
            train_idx=all_idx,
            test_idx=all_idx
        )

        append_result(
            metrics=metrics,
            regime="all_all",
            repeat=0,
            requested_train_amount=1.0,
            train_amount_type="fraction",
            training_pool_size=len(all_idx),
            test_scope="all"
        )

        # ====================================================
        # Random train/test splits
        # ====================================================

        for amount in random_train_amounts:
            n_train, amount_type = (
                resolve_random_train_amount(
                    amount=amount,
                    n_total=len(all_idx)
                )
            )

            for repeat in range(n_repeats):
                shuffled = rng.permutation(all_idx)

                train_idx = shuffled[:n_train]
                test_idx = shuffled[n_train:]

                _, metrics = fit_and_evaluate(
                    X=X,
                    activity=activity,
                    labels=labels,
                    train_idx=train_idx,
                    test_idx=test_idx
                )

                append_result(
                    metrics=metrics,
                    regime="random",
                    repeat=repeat,
                    requested_train_amount=amount,
                    train_amount_type=amount_type,
                    training_pool_size=len(all_idx),
                    test_scope="held_out_complement"
                )

        # ====================================================
        # Mutation extrapolation
        # ====================================================

        if mutation_ks is None:
            current_ks = np.sort(
                np.unique(mut_order[valid])
            )[:-1]
        else:
            current_ks = np.asarray(mutation_ks)

        current_ks = current_ks[current_ks > 0]

        for k in current_ks:
            train_pool = np.where(
                valid & (mut_order <= k)
            )[0]

            all_test_idx = np.where(
                valid & (mut_order > k)
            )[0]

            if (
                len(train_pool) == 0 or
                len(all_test_idx) < 2
            ):
                continue

            exact_test_orders = np.sort(
                np.unique(
                    mut_order[all_test_idx]
                )
            )

            for fraction in mutation_train_fractions:
                if not 0 < fraction <= 1:
                    raise ValueError(
                        "mutation_train_fractions must "
                        "be between 0 and 1."
                    )

                n_train = int(
                    round(len(train_pool) * fraction)
                )
                n_train = np.clip(
                    n_train,
                    1,
                    len(train_pool)
                )

                repeats = (
                    1
                    if n_train == len(train_pool)
                    else n_repeats
                )

                for repeat in range(repeats):
                    train_idx = rng.choice(
                        train_pool,
                        size=n_train,
                        replace=False
                    )

                    # Fit once
                    model = LinearRegression()
                    model.fit(
                        X[train_idx],
                        activity[train_idx]
                    )

                    # Predict once on all variants above K
                    all_predictions = model.predict(
                        X[all_test_idx]
                    )

                    # ----------------------------------------
                    # All test mutation orders pooled together
                    # ----------------------------------------

                    metrics = score_predictions(
                        activity=activity,
                        labels=labels,
                        train_idx=train_idx,
                        test_idx=all_test_idx,
                        predictions=all_predictions
                    )

                    append_result(
                        metrics=metrics,
                        regime="mutation",
                        repeat=repeat,
                        requested_train_amount=fraction,
                        train_amount_type="fraction",
                        training_pool_size=len(train_pool),
                        k=int(k),
                        test_scope="all_above_k",
                        test_order=np.nan
                    )

                    # ----------------------------------------
                    # Each exact test mutation order separately
                    # ----------------------------------------

                    for test_order in exact_test_orders:
                        order_mask = (
                            mut_order[all_test_idx] ==
                            test_order
                        )

                        exact_test_idx = all_test_idx[
                            order_mask
                        ]

                        exact_predictions = all_predictions[
                            order_mask
                        ]

                        metrics = score_predictions(
                            activity=activity,
                            labels=labels,
                            train_idx=train_idx,
                            test_idx=exact_test_idx,
                            predictions=exact_predictions
                        )

                        append_result(
                            metrics=metrics,
                            regime="mutation",
                            repeat=repeat,
                            requested_train_amount=fraction,
                            train_amount_type="fraction",
                            training_pool_size=len(train_pool),
                            k=int(k),
                            test_scope="exact_order",
                            test_order=int(test_order)
                        )

    results_df = pd.DataFrame(rows)

    if classification_error_frames:
        classification_errors_df = pd.concat(
            classification_error_frames,
            ignore_index=True
        )
    else:
        classification_errors_df = pd.DataFrame(
            columns=[
                "dataset",
                "mutations_in_train",
                "positive_indices_train",
                "negative_indices_train",
                "false_positive_indices_test",
                "false_negative_indices_test",
                "test_order"
            ]
        )

    return results_df, classification_errors_df


def run_epistasis_experiments(
    datasets,
    percentile=0.50,
    max_mutations=10,
    calculate_non_epistatic_reverses=False
):
    """
    Identify sign epistasis from binary activity labels.

    Expected double-mutant phenotype
    ---------------------------------
    positive + positive -> positive
    positive + negative -> negative
    negative + negative -> negative

    Epistasis types
    ----------------
    positive + positive -> negative:
        negative_sign_epistasis

    positive + negative -> positive:
        weak_positive_sign_epistasis

    negative + negative -> positive:
        strong_positive_sign_epistasis

    A reverse is a higher-order variant containing both exact mutations
    whose label is opposite to the observed double-mutant label.

    For epistatic pairs, this means returning to the phenotype expected from
    the two single-mutant labels. Higher-order variants containing the pair
    whose label remains unchanged are recorded as non-reversed.
    """
    result_rows = []

    for dataset_name in datasets:
        print(f"Working on {dataset_name}")

        df = pd.read_csv(
            DATASET_PATHS[dataset_name]
        )

        if "num_muts" in df.columns:
            mutation_column = "num_muts"
        else:
            mutation_column = num_muts_column_name[
                dataset_name
            ]

        df = df[
            df[mutation_column] <= max_mutations
        ].copy()

        # Preserve indices from the original CSV
        df["_source_index"] = df.index
        df = df.reset_index(drop=True)

        source_indices = df[
            "_source_index"
        ].to_numpy()

        mut_order = df[
            mutation_column
        ].to_numpy()

        relevant_columns = get_relevant_columns(
            dataset_name,
            df
        )

        sequence_matrix = df[
            relevant_columns
        ].astype(str).to_numpy()

        # ----------------------------------------------------
        # Create binary labels
        # ----------------------------------------------------

        if dataset_name == "gfp":
            activity = (
                1 - df["inactive"]
            ).to_numpy(dtype=float)

            valid = np.isfinite(activity)
            labels = np.zeros(
                len(activity),
                dtype=int
            )
            labels[valid] = activity[
                valid
            ].astype(int)

        else:
            activity_column = datasets_and_activity[
                dataset_name
            ]["activity_col"]

            activity = df[
                activity_column
            ].to_numpy(dtype=float)

            labels, _, valid = make_activity_labels(
                activity,
                percentile
            )

        # ----------------------------------------------------
        # Identify the WT sequence
        # ----------------------------------------------------

        wt_rows = np.where(
            mut_order == 0
        )[0]

        if len(wt_rows) == 0:
            raise ValueError(
                f"{dataset_name}: no WT row was found."
            )

        wt_sequence = sequence_matrix[
            wt_rows[0]
        ]

        # ----------------------------------------------------
        # Identify all observed single mutations and labels
        # ----------------------------------------------------

        single_labels = defaultdict(set)
        single_indices = defaultdict(list)

        single_rows = np.where(
            valid & (mut_order == 1)
        )[0]

        for row_idx in single_rows:
            changed_positions = np.where(
                sequence_matrix[row_idx] != wt_sequence
            )[0]

            if len(changed_positions) != 1:
                continue

            position_idx = int(
                changed_positions[0]
            )

            mutation_id = (
                position_idx,
                str(wt_sequence[position_idx]),
                str(sequence_matrix[row_idx, position_idx])
            )

            single_labels[mutation_id].add(
                int(labels[row_idx])
            )
            single_indices[mutation_id].append(
                source_indices[row_idx]
            )

        # Conflicting labels would make the expected phenotype ambiguous
        for mutation_id, observed_labels in single_labels.items():
            if len(observed_labels) > 1:
                raise ValueError(
                    f"{dataset_name}: single mutation "
                    f"{mutation_id} has conflicting labels "
                    f"{observed_labels}."
                )

        mutations = sorted(
            single_labels,
            key=lambda mutation: (
                mutation[0],
                mutation[2]
            )
        )

        mutation_to_column = {
            mutation: column_idx
            for column_idx, mutation in enumerate(mutations)
        }

        mutation_label = {
            mutation: next(iter(single_labels[mutation]))
            for mutation in mutations
        }

        # ----------------------------------------------------
        # Build sparse row × mutation incidence matrix
        # ----------------------------------------------------

        incidence_rows = []
        incidence_columns = []

        mutations_by_position = defaultdict(list)

        for mutation in mutations:
            mutations_by_position[
                mutation[0]
            ].append(mutation)

        for position_idx, position_mutations in (
            mutations_by_position.items()
        ):
            position_values = sequence_matrix[
                :,
                position_idx
            ]

            for mutation in position_mutations:
                mutant_aa = mutation[2]

                matching_rows = np.where(
                    position_values == mutant_aa
                )[0]

                incidence_rows.extend(
                    matching_rows.tolist()
                )
                incidence_columns.extend(
                    [mutation_to_column[mutation]] *
                    len(matching_rows)
                )

        incidence = sparse.coo_matrix(
            (
                np.ones(
                    len(incidence_rows),
                    dtype=np.int8
                ),
                (
                    incidence_rows,
                    incidence_columns
                )
            ),
            shape=(
                len(df),
                len(mutations)
            )
        ).tocsr()

        # CSC allows efficient lookup of every variant containing
        # a particular mutation.
        incidence_csc = incidence.tocsc()

        # ----------------------------------------------------
        # Find observed double mutants
        # ----------------------------------------------------

        pair_to_rows = defaultdict(list)

        double_rows = np.where(
            valid & (mut_order == 2)
        )[0]

        for row_idx in double_rows:
            start = incidence.indptr[row_idx]
            end = incidence.indptr[row_idx + 1]

            mutation_columns = incidence.indices[
                start:end
            ]

            # Ignore doubles containing a mutation without a
            # corresponding observed single mutant.
            if len(mutation_columns) != 2:
                continue

            pair = tuple(
                sorted(mutation_columns.tolist())
            )
            pair_to_rows[pair].append(row_idx)

        # ----------------------------------------------------
        # Classify epistasis and search for reverses
        # ----------------------------------------------------

        reverse_orders = range(
            3,
            max_mutations + 1
        )

        for pair, pair_rows in pair_to_rows.items():
            mutation_1 = mutations[pair[0]]
            mutation_2 = mutations[pair[1]]

            single_1_label = mutation_label[
                mutation_1
            ]
            single_2_label = mutation_label[
                mutation_2
            ]

            # Product/AND rule
            expected_label = int(
                single_1_label == 1 and
                single_2_label == 1
            )

            pair_labels = np.unique(
                labels[pair_rows]
            )

            if len(pair_labels) > 1:
                raise ValueError(
                    f"{dataset_name}: pair "
                    f"{mutation_1}, {mutation_2} has "
                    f"conflicting double-mutant labels."
                )

            actual_label = int(
                pair_labels[0]
            )

            if expected_label == 1 and actual_label == 0:
                epistasis_type = "negative_sign_epistasis"

            elif expected_label == 0 and actual_label == 1:
                if (
                    single_1_label == 0 and
                    single_2_label == 0
                ):
                    epistasis_type = (
                        "strong_positive_sign_epistasis"
                    )
                else:
                    epistasis_type = (
                        "weak_positive_sign_epistasis"
                    )

            elif expected_label == 1:
                epistasis_type = (
                    "no_epistasis_expected_positive"
                )

            else:
                epistasis_type = (
                    "no_epistasis_expected_negative"
                )

            is_epistatic = (
                expected_label != actual_label
            )

            calculate_reverses = (
                is_epistatic or
                calculate_non_epistatic_reverses
            )

            if calculate_reverses:
                mutation_1_rows = incidence_csc.indices[
                    incidence_csc.indptr[pair[0]]:
                    incidence_csc.indptr[pair[0] + 1]
                ]

                mutation_2_rows = incidence_csc.indices[
                    incidence_csc.indptr[pair[1]]:
                    incidence_csc.indptr[pair[1] + 1]
                ]

                # Variants containing both exact substitutions
                containing_pair = np.intersect1d(
                    mutation_1_rows,
                    mutation_2_rows,
                    assume_unique=True
                )

                higher_order_rows = containing_pair[
                    valid[containing_pair] &
                    (mut_order[containing_pair] > 2)
                ]

                # A reverse is opposite to the observed double-mutant
                # phenotype; all remaining variants are non-reversed.
                reverse_rows = higher_order_rows[
                    labels[higher_order_rows] != actual_label
                ]
                non_reverse_rows = higher_order_rows[
                    labels[higher_order_rows] == actual_label
                ]

                reverses_by_order = {}
                non_reverses_by_order = {}
                reverse_indices = {}
                non_reverse_indices = {}

                for order in reverse_orders:
                    order_reverse_rows = reverse_rows[
                        mut_order[reverse_rows] == order
                    ]
                    order_non_reverse_rows = non_reverse_rows[
                        mut_order[non_reverse_rows] == order
                    ]

                    reverses_by_order[int(order)] = int(
                        len(order_reverse_rows)
                    )
                    non_reverses_by_order[int(order)] = int(
                        len(order_non_reverse_rows)
                    )

                    reverse_indices[int(order)] = (
                        source_indices[order_reverse_rows].tolist()
                    )
                    non_reverse_indices[int(order)] = (
                        source_indices[order_non_reverse_rows].tolist()
                    )

            else:
                reverses_by_order = {}
                non_reverses_by_order = {}
                reverse_indices = {}
                non_reverse_indices = {}

            position_1 = relevant_columns[
                mutation_1[0]
            ]
            position_2 = relevant_columns[
                mutation_2[0]
            ]

            mutation_1_name = (
                f"{position_1}:"
                f"{mutation_1[1]}>{mutation_1[2]}"
            )
            mutation_2_name = (
                f"{position_2}:"
                f"{mutation_2[1]}>{mutation_2[2]}"
            )

            result_rows.append({
                "dataset": dataset_name,
                "pair_identity": (
                    mutation_1_name,
                    mutation_2_name
                ),
                "single_1_status": (
                    "positive"
                    if single_1_label == 1
                    else "negative"
                ),
                "single_2_status": (
                    "positive"
                    if single_2_label == 1
                    else "negative"
                ),
                "expected_phenotype": (
                    "positive"
                    if expected_label == 1
                    else "negative"
                ),
                "actual_phenotype": (
                    "positive"
                    if actual_label == 1
                    else "negative"
                ),
                "epistasis_type": epistasis_type,
                "is_epistatic": is_epistatic,
                "pair_indices": (
                    source_indices[pair_rows].tolist()
                ),
                "reverses_mutational_order": (
                    reverses_by_order
                ),
                "non_reverses_mutational_order": (
                    non_reverses_by_order
                ),
                "reverse_indices": reverse_indices,
                "non_reverse_indices": non_reverse_indices
            })

    return pd.DataFrame(result_rows)


output_dir = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/data/merged_datasets/linear_baslines"

datasets_to_run = [
    "casp",
    "his2",
    "his5",
    "pard3",
    "gcn4",
    "lov",
    "gfp",
    # "aamyl",
    "pte",
    "nmt",
    # "ired",
    # "nucb",
    # "trpb"
]

# epistasis_results = run_epistasis_experiments(
#     datasets=datasets_to_run,
#     percentile=0.50,
#     max_mutations=11,
#     calculate_non_epistatic_reverses=True
# )
# epistasis_results.to_csv(
#     os.path.join(output_dir, f"{datasets_to_run[0]}_epis.csv"),
#     index=False
# )


results, _ = run_linear_experiments(
    datasets=datasets_to_run,
    percentile=0.50,

    # Values <=1 are fractions; values >1 are sequence counts
    random_train_amounts=[
        10,
        20,
        50,
        100,
        1000,
        5000
    ],

    # Mutation experiment still uses fractions of the <=K pool
    mutation_train_fractions=(1.0,),

    mutation_ks=None,
    classification_train_ks=range(1, 7),
    collect_classification_errors=False,
    n_repeats=10,
    max_mutations=10,
    random_seed=0
)





results.to_csv(
    os.path.join(output_dir, f"{datasets_to_run[0]}_tmp.csv"),
    index=False
)
# mutation_classification_errors.to_csv(
#     os.path.join(
#         output_dir,
#         f"{datasets_to_run[0]}_mutation_classification_errors.csv"
#     ),
#     index=False
# )

# print(results)
# print(mutation_classification_errors)
