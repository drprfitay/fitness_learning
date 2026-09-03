#!/usr/bin/env python3
"""Bootstrap onehot-vs-feature geometry and linear reconstruction checks."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_PT = Path("features.pt")
SAVE_CSV = None
ONEHOT_KEY = "onehot"
SKIP_KEYS = {"activity"}

N = 512
K = 100
SEED = 0
CV_FOLDS = 5
INNER_CV_FOLDS = 5
ALPHAS = np.logspace(-4, 4, 25)
N_JOBS = -1

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


def geometry_spearman(ohe, feat):
    feat_norm = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    cosine_similarity_matrix = feat_norm @ feat_norm.T

    flat_onehot = ohe.reshape(len(ohe), -1)
    hamming_dist_matrix = squareform(pdist(flat_onehot, metric="hamming"))

    idx = np.triu_indices_from(cosine_similarity_matrix, k=1)
    cosine = cosine_similarity_matrix[idx]
    hamming = hamming_dist_matrix[idx]

    keep = np.isfinite(cosine) & np.isfinite(hamming)
    if keep.sum() < 2:
        return float("nan")
    cosine = cosine[keep]
    hamming = hamming[keep]
    if np.std(cosine) == 0 or np.std(hamming) == 0:
        return float("nan")
    return float(spearmanr(cosine, hamming).correlation)


def global_r2(y, y_hat):
    numerator = ((y - y_hat) ** 2).sum()
    denominator = ((y - y.mean(axis=0)) ** 2).sum()
    return float(1.0 - numerator / denominator) if denominator > 0 else float("nan")


def reconstruction_r2(x, y):
    outer_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
    inner_cv = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=1)
    y_oof = np.empty_like(y, dtype=float)

    for train_idx, test_idx in outer_cv.split(x):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge()),
            ]
        )
        search = GridSearchCV(
            model,
            param_grid={"ridge__alpha": ALPHAS},
            scoring="neg_mean_squared_error",
            cv=inner_cv,
            n_jobs=N_JOBS,
        )
        search.fit(x[train_idx], y[train_idx])
        y_oof[test_idx] = search.predict(x[test_idx])

    feature_r2 = r2_score(y, y_oof, multioutput="raw_values")
    return {
        "global_r2": global_r2(y, y_oof),
        "weighted_r2": float(r2_score(y, y_oof, multioutput="variance_weighted")),
        "mean_feature_r2": float(np.mean(feature_r2)),
        "median_feature_r2": float(np.median(feature_r2)),
        "feature_r2_p0": float(np.percentile(feature_r2, 0)),
        "feature_r2_p5": float(np.percentile(feature_r2, 5)),
        "feature_r2_p25": float(np.percentile(feature_r2, 25)),
        "feature_r2_p50": float(np.percentile(feature_r2, 50)),
        "feature_r2_p75": float(np.percentile(feature_r2, 75)),
        "feature_r2_p95": float(np.percentile(feature_r2, 95)),
        "feature_r2_p100": float(np.percentile(feature_r2, 100)),
    }


def one_round(ohe, feat, sample):
    ohe_sample = ohe[sample]
    feat_sample = feat[sample]
    row = {
        "geometry_spearman": geometry_spearman(ohe_sample, feat_sample),
    }
    for prefix, x, y in (
        ("ohe_to_feature", ohe_sample, feat_sample),
        ("feature_to_ohe", feat_sample, ohe_sample),
    ):
        for key, value in reconstruction_r2(x, y).items():
            row[f"{prefix}_{key}"] = value
    return row


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return np.mean(values), np.std(values)


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
    header = "feature  geom_rho  ohe->feat_r2  feat->ohe_r2"
    print(header)
    print("-" * len(header))
    for row in summary_rows(results):
        print(
            f"{row['feature']:<8} "
            f"{row['geometry_spearman_mean']:>7.3f}+/-{row['geometry_spearman_std']:<6.3f} "
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
    if N > len(ohe):
        raise ValueError(f"N={N} is larger than row count {len(ohe)}")
    if N < 3:
        raise ValueError("N must be at least 3")
    if N < CV_FOLDS or N - int(np.ceil(N / CV_FOLDS)) < INNER_CV_FOLDS:
        raise ValueError("N is too small for CV_FOLDS and INNER_CV_FOLDS")

    rng = np.random.default_rng(SEED)
    feature_keys = [k for k in arrays if k != ONEHOT_KEY]
    if not feature_keys:
        raise ValueError(f"no feature keys besides {ONEHOT_KEY!r}")
    results = {k: [] for k in feature_keys}

    for i in range(1, K + 1):
        sample = rng.choice(len(ohe), size=N, replace=False)

        for key in feature_keys:
            results[key].append(one_round(ohe, arrays[key], sample))
        if VERBOSE and (i == 1 or i % PRINT_EVERY == 0 or i == K):
            print(f"\nround {i}/{K}")
            print_summary(results)

    print("\nfinal")
    print_summary(results)
    save_summary(results)


if __name__ == "__main__":
    main()
