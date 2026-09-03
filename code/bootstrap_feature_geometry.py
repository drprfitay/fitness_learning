#!/usr/bin/env python3
"""Bootstrap onehot-vs-feature geometry and linear reconstruction checks."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_PT = Path("features.pt")
SAVE_CSV = None
ONEHOT_KEY = "onehot"
SKIP_KEYS = {"activity"}

N = 512
K = 100
TEST_FRAC = 0.5
SEED = 0

ALPHAS = np.logspace(-4, 4, 25)
RIDGE_CV_FOLDS = 5

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


def global_r2(y, y_hat):
    numerator = ((y - y_hat) ** 2).sum()
    denominator = ((y - y.mean(axis=0)) ** 2).sum()
    return float(1.0 - numerator / denominator) if denominator > 0 else float("nan")


def reconstruction_r2(x, y, train_idx, test_idx):
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, cv=RIDGE_CV_FOLDS),
    )
    model.fit(x[train_idx], y[train_idx])
    y_hat = model.predict(x[test_idx])
    y_test = y[test_idx]
    return {
        "global_r2": global_r2(y_test, y_hat),
        "weighted_r2": float(r2_score(y_test, y_hat, multioutput="variance_weighted")),
    }


def one_round(ohe, feat, sample, train_idx, test_idx):
    row = {
        "geometry_distance_spearman": geometry_distance_spearman(ohe[sample], feat[sample]),
    }
    for prefix, x, y in (
        ("ohe_to_feature", ohe, feat),
        ("feature_to_ohe", feat, ohe),
    ):
        for key, value in reconstruction_r2(x, y, train_idx, test_idx).items():
            row[f"{prefix}_{key}"] = value
    return row


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
            f"{row['geometry_distance_spearman_mean']:>7.3f}+/-{row['geometry_distance_spearman_std']:<6.3f} "
            f"{row['ohe_to_feature_weighted_r2_mean']:>7.3f}+/-{row['ohe_to_feature_weighted_r2_std']:<6.3f} "
            f"{row['feature_to_ohe_weighted_r2_mean']:>7.3f}+/-{row['feature_to_ohe_weighted_r2_std']:<6.3f}"
        )


def save_summary(results):
    out = Path(SAVE_CSV).expanduser() if SAVE_CSV else Path(FEATURES_PT).expanduser().with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows(results)).to_csv(out, index=False)
    print(f"\nsaved {out}")


def main():
    arrays = load_features(FEATURES_PT)
    ohe = arrays[ONEHOT_KEY]
    train_size = int(round(N * (1.0 - TEST_FRAC)))

    if N > len(ohe):
        raise ValueError(f"N={N} is larger than row count {len(ohe)}")
    if N < 3:
        raise ValueError("N must be at least 3")
    if train_size == 0 or train_size == N:
        raise ValueError("TEST_FRAC leaves an empty train or test split")
    if train_size < RIDGE_CV_FOLDS:
        raise ValueError("train split is smaller than RIDGE_CV_FOLDS")

    rng = np.random.default_rng(SEED)
    feature_keys = [k for k in arrays if k != ONEHOT_KEY]
    if not feature_keys:
        raise ValueError(f"no feature keys besides {ONEHOT_KEY!r}")
    results = {k: [] for k in feature_keys}

    for i in range(1, K + 1):
        sample = rng.choice(len(ohe), size=N, replace=False)
        rng.shuffle(sample)
        train_idx = sample[:train_size]
        test_idx = sample[train_size:]

        for key in feature_keys:
            results[key].append(one_round(ohe, arrays[key], sample, train_idx, test_idx))
        if VERBOSE and (i == 1 or i % PRINT_EVERY == 0 or i == K):
            print(f"\nround {i}/{K}")
            print_summary(results)

    print("\nfinal")
    print_summary(results)
    save_summary(results)


if __name__ == "__main__":
    main()
