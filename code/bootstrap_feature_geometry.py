#!/usr/bin/env python3
"""Bootstrap onehot-vs-feature geometry and linear reconstruction checks."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_PT = Path("features.pt")
SAVE_CSV = None
DATASET_NAME = None
ONEHOT_KEY = "onehot"
SKIP_KEYS = {"activity"}

N = 512
K = 100
TEST_FRAC = 0.5
SEED = 0

ALPHAS = np.logspace(-2, 4, 13)
RIDGE_CV_FOLDS = 5
USE_FIXED_ALPHA = False
FIXED_ALPHA = 100.0
SUPPRESS_LINALG_WARNINGS = True
PRINT_SELECTED_ALPHA = True
REPORT_OHE_TO_FEATURE_R2_QUANTILES = False
R2_QUANTILES = [0, 5, 25, 50, 75, 95, 100]

SAVE_RECONSTRUCTION_EXAMPLES = False
RECONSTRUCTION_DIRNAME = "reconstruction_analysis"
HEATMAP_MAX_VARIANTS = 200
HEATMAP_MAX_FEATURES = 300

VERBOSE = True
PRINT_EVERY = 5


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(len(x), -1)


def load_features(path):
    import torch

    data = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if ONEHOT_KEY not in data:
        raise KeyError(f"missing {ONEHOT_KEY!r} in {path}")

    arrays = {k: to_numpy(v) for k, v in data.items() if k not in SKIP_KEYS}
    rows = {k: x.shape[0] for k, x in arrays.items()}
    if len(set(rows.values())) != 1:
        raise ValueError(f"not all features have the same row count: {rows}")
    for k, x in arrays.items():
        if not np.isfinite(x).all():
            raise ValueError(f"{k!r} has nan/inf values")
    return arrays


def geometry_distance_spearman(ohe, feat):
    feat_norm = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    cosine_similarity_matrix = feat_norm @ feat_norm.T

    flat_onehot = ohe.reshape(len(ohe), -1)
    hamming_dist_matrix = squareform(pdist(flat_onehot, metric="hamming"))

    idx = np.triu_indices_from(cosine_similarity_matrix, k=1)
    cosine_distance = 1.0 - cosine_similarity_matrix[idx]
    hamming_distance = hamming_dist_matrix[idx]

    keep = np.isfinite(cosine_distance) & np.isfinite(hamming_distance)
    if keep.sum() < 2:
        return float("nan")
    cosine_distance = cosine_distance[keep]
    hamming_distance = hamming_distance[keep]
    if np.std(cosine_distance) == 0 or np.std(hamming_distance) == 0:
        return float("nan")
    return float(spearmanr(cosine_distance, hamming_distance).correlation)


def pairwise_matrices(ohe, feat):
    feat_norm = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    cosine_similarity = feat_norm @ feat_norm.T
    hamming_distance = squareform(pdist(ohe.reshape(len(ohe), -1), metric="hamming"))
    return {
        "embedding_cosine_similarity": cosine_similarity,
        "embedding_cosine_distance": 1.0 - cosine_similarity,
        "onehot_hamming_distance": hamming_distance,
        "onehot_hamming_similarity": 1.0 - hamming_distance,
    }


def global_r2(y, y_hat):
    numerator = ((y - y_hat) ** 2).sum()
    denominator = ((y - y.mean(axis=0)) ** 2).sum()
    return float(1.0 - numerator / denominator) if denominator > 0 else float("nan")


def make_ridge_model():
    if USE_FIXED_ALPHA:
        return make_pipeline(StandardScaler(), Ridge(alpha=FIXED_ALPHA))
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=RIDGE_CV_FOLDS))


def model_alpha(model):
    if USE_FIXED_ALPHA:
        return FIXED_ALPHA
    return model.named_steps["ridgecv"].alpha_


def reconstruction_r2(x, y, train_idx, test_idx, label, report_quantiles=False, return_example=False):
    model = make_ridge_model()
    if SUPPRESS_LINALG_WARNINGS:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            model.fit(x[train_idx], y[train_idx])
    else:
        model.fit(x[train_idx], y[train_idx])

    alpha = model_alpha(model)
    if VERBOSE and PRINT_SELECTED_ALPHA:
        print(f"  {label} alpha: {alpha}")

    y_hat = model.predict(x[test_idx])
    y_test = y[test_idx]
    scores = {
        "global_r2": global_r2(y_test, y_hat),
        "weighted_r2": float(r2_score(y_test, y_hat, multioutput="variance_weighted")),
    }
    if report_quantiles:
        per_feature_r2 = r2_score(y_test, y_hat, multioutput="raw_values")
        for q in R2_QUANTILES:
            scores[f"per_feature_r2_q{q}"] = float(np.percentile(per_feature_r2, q))
    if return_example:
        return scores, alpha, y_test, y_hat
    return scores, alpha


def one_round(ohe, feat, sample, train_idx, test_idx, feature_name, return_example=False):
    row = {
        "geometry_distance_spearman": geometry_distance_spearman(ohe[sample], feat[sample]),
    }
    example = None
    for prefix, x, y in (
        ("ohe_to_feature", ohe, feat),
        ("feature_to_ohe", feat, ohe),
    ):
        label = f"{feature_name} {prefix}"
        report_quantiles = prefix == "ohe_to_feature" and REPORT_OHE_TO_FEATURE_R2_QUANTILES
        result = reconstruction_r2(x, y, train_idx, test_idx, label, report_quantiles, return_example)
        if return_example:
            scores, alpha, y_test, y_hat = result
            if example is None:
                example = {"sample_idx": sample, "test_idx": test_idx}
            example[prefix] = {"actual": y_test, "pred": y_hat}
        else:
            scores, alpha = result
        row[f"{prefix}_alpha"] = alpha
        for key, value in scores.items():
            row[f"{prefix}_{key}"] = value
    return row, example


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def summary_rows(results):
    rows = []
    for key, values in results.items():
        row = {"feature": key}
        for metric in values[0]:
            if metric.endswith("_alpha"):
                continue
            mean, std = summarize([r[metric] for r in values])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return rows


def print_summary(results):
    header = "feature  geom_dist_rho  ohe->feat_r2  feat->ohe_r2"
    print(header)
    print("-" * len(header))
    for row in summary_rows(results):
        print(
            f"{row['feature']:<8} "
            f"{row['geometry_distance_spearman_mean']:>7.3f} "
            f"{row['ohe_to_feature_weighted_r2_mean']:>7.3f} "
            f"{row['feature_to_ohe_weighted_r2_mean']:>7.3f}"
        )


def metric_alpha(metric, row):
    if metric.startswith("ohe_to_feature_"):
        return row["ohe_to_feature_alpha"]
    if metric.startswith("feature_to_ohe_"):
        return row["feature_to_ohe_alpha"]
    return float("nan")


def output_metric_name(metric):
    if metric == "geometry_distance_spearman":
        return "geom_rho"
    if metric == "ohe_to_feature_weighted_r2":
        return "ohe_feat_r2"
    if metric == "feature_to_ohe_weighted_r2":
        return "feat_ohe_r2"
    if metric == "ohe_to_feature_global_r2":
        return "ohe_feat_global_r2"
    if metric == "feature_to_ohe_global_r2":
        return "feat_ohe_global_r2"
    if metric.startswith("ohe_to_feature_per_feature_r2_q"):
        return metric.replace("ohe_to_feature_per_feature_r2_q", "ohe_feat_r2_q")
    return metric


def dataset_name():
    if DATASET_NAME is not None:
        return DATASET_NAME
    parent = Path(FEATURES_PT).expanduser().parent
    return parent.name if parent.name else Path(FEATURES_PT).expanduser().stem


def iteration_rows(results, train_size, test_size):
    rows = []
    dataset = dataset_name()
    for feature, feature_results in results.items():
        for iteration, result in enumerate(feature_results, start=1):
            for metric, value in result.items():
                if metric.endswith("_alpha"):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "iteration": iteration,
                        "feature": feature,
                        "metric": output_metric_name(metric),
                        "alpha": metric_alpha(metric, result),
                        "train_size": train_size,
                        "test_size": test_size,
                        "value": value,
                    }
                )
    return rows


def save_results(results, train_size, test_size):
    out = Path(SAVE_CSV).expanduser() if SAVE_CSV else Path(FEATURES_PT).expanduser().with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(iteration_rows(results, train_size, test_size)).to_csv(out, index=False)
    print(f"\nsaved {out}")


def ordered_indices(x, axis):
    n = x.shape[axis]
    if n <= 1:
        return np.arange(n)
    values = x if axis == 0 else x.T
    try:
        return leaves_list(linkage(values, method="average", metric="euclidean"))
    except ValueError:
        return np.arange(n)


def heatmap_order(x):
    row_idx = np.arange(x.shape[0])
    col_idx = np.arange(x.shape[1])
    if len(row_idx) > HEATMAP_MAX_VARIANTS:
        row_idx = row_idx[:HEATMAP_MAX_VARIANTS]
    if len(col_idx) > HEATMAP_MAX_FEATURES:
        col_idx = np.argsort(np.nanvar(x, axis=0))[-HEATMAP_MAX_FEATURES:]

    view = x[np.ix_(row_idx, col_idx)]
    row_idx = row_idx[ordered_indices(view, axis=0)]
    col_idx = col_idx[ordered_indices(view, axis=1)]
    return row_idx, col_idx


def heatmap_limit(x):
    limit = np.nanpercentile(np.abs(x), 99)
    if not np.isfinite(limit) or limit == 0:
        return 1.0
    return limit


def save_ohe_to_feature_heatmap(feature_dir, feature_name, actual, pred):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_idx, col_idx = heatmap_order(actual)
    actual_view = actual[np.ix_(row_idx, col_idx)]
    pred_view = pred[np.ix_(row_idx, col_idx)]
    error_view = pred_view - actual_view
    vmax = heatmap_limit(np.r_[actual_view.ravel(), pred_view.ravel()])
    err_vmax = heatmap_limit(error_view)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, data, title, limit in (
        (axes[0], actual_view, "actual", vmax),
        (axes[1], pred_view, "pred", vmax),
        (axes[2], error_view, "pred - actual", err_vmax),
    ):
        im = ax.imshow(data, aspect="auto", cmap="viridis" if title != "pred - actual" else "coolwarm", vmin=-limit if title == "pred - actual" else None, vmax=limit)
        ax.set_title(title)
        ax.set_xlabel("features")
        ax.set_ylabel("variants")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{feature_name} ohe -> feature")
    fig.savefig(feature_dir / "ohe_to_feature_heatmap.png", dpi=180)
    plt.close(fig)


def save_reconstruction_example(feature_name, ohe, feat, example):
    base = Path(FEATURES_PT).expanduser().parent / RECONSTRUCTION_DIRNAME / feature_name
    base.mkdir(parents=True, exist_ok=True)

    sample_idx = example["sample_idx"]
    test_idx = example["test_idx"]
    np.save(base / "sample_indices.npy", sample_idx)
    np.save(base / "test_indices.npy", test_idx)
    np.save(base / "sample_original_onehot.npy", ohe[sample_idx])
    np.save(base / "sample_original_embedding.npy", feat[sample_idx])
    np.save(base / "original_onehot.npy", ohe[test_idx])
    np.save(base / "original_embedding.npy", feat[test_idx])

    for name, matrix in pairwise_matrices(ohe[sample_idx], feat[sample_idx]).items():
        np.save(base / f"sample_pairwise_{name}.npy", matrix)

    for direction in ("ohe_to_feature", "feature_to_ohe"):
        np.save(base / f"{direction}_actual.npy", example[direction]["actual"])
        np.save(base / f"{direction}_pred.npy", example[direction]["pred"])

    save_ohe_to_feature_heatmap(
        base,
        feature_name,
        example["ohe_to_feature"]["actual"],
        example["ohe_to_feature"]["pred"],
    )


def main():
    arrays = load_features(FEATURES_PT)
    ohe = arrays[ONEHOT_KEY]
    sample_size = len(ohe) if N == -1 else N
    train_size = int(round(sample_size * (1.0 - TEST_FRAC)))

    if sample_size > len(ohe):
        raise ValueError(f"N={N} is larger than row count {len(ohe)}")
    if sample_size < 3:
        raise ValueError("N must be -1 or at least 3")
    if train_size == 0 or train_size == sample_size:
        raise ValueError("TEST_FRAC leaves an empty train or test split")
    if not USE_FIXED_ALPHA and train_size < RIDGE_CV_FOLDS:
        raise ValueError("train split is smaller than RIDGE_CV_FOLDS")

    rng = np.random.default_rng(SEED)
    feature_keys = [k for k in arrays if k != ONEHOT_KEY]
    if not feature_keys:
        raise ValueError(f"no feature keys besides {ONEHOT_KEY!r}")
    results = {k: [] for k in feature_keys}

    for i in range(1, K + 1):
        sample = rng.choice(len(ohe), size=sample_size, replace=False)
        rng.shuffle(sample)
        train_idx = sample[:train_size]
        test_idx = sample[train_size:]

        if VERBOSE and PRINT_SELECTED_ALPHA:
            print(f"\nround {i}/{K} selected alphas")
        for key in feature_keys:
            row, example = one_round(
                ohe,
                arrays[key],
                sample,
                train_idx,
                test_idx,
                key,
                return_example=SAVE_RECONSTRUCTION_EXAMPLES and i == 1,
            )
            results[key].append(row)
            if example is not None:
                save_reconstruction_example(key, ohe, arrays[key], example)
        if VERBOSE and (i == 1 or i % PRINT_EVERY == 0 or i == K):
            print(f"\nround {i}/{K}")
            print_summary(results)

    print("\nfinal")
    print_summary(results)
    save_results(results, train_size, sample_size - train_size)


if __name__ == "__main__":
    main()
