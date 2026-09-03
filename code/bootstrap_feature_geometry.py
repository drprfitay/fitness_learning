#!/usr/bin/env python3
"""Bootstrap onehot-vs-feature geometry and linear reconstruction checks."""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


FEATURES_PT = Path("features.pt")
ONEHOT_KEY = "onehot"
SKIP_KEYS = {"activity"}

N = 512
K = 100
TEST_FRAC = 0.5
RIDGE_ALPHA = 1.0
SEED = 0

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
    ohe_hamming = pdist(ohe, metric="hamming")
    feat_cosine = 1.0 - pdist(feat, metric="cosine")
    keep = np.isfinite(ohe_hamming) & np.isfinite(feat_cosine)
    if keep.sum() < 2:
        return float("nan")
    ohe_hamming = ohe_hamming[keep]
    feat_cosine = feat_cosine[keep]
    if np.std(ohe_hamming) == 0 or np.std(feat_cosine) == 0:
        return float("nan")
    return float(spearmanr(ohe_hamming, feat_cosine).correlation)


def reconstruction_nmse(x, y, train_idx, test_idx):
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    mse = np.mean((pred - y[test_idx]) ** 2)
    var = np.mean((y[test_idx] - y[test_idx].mean(axis=0)) ** 2)
    return float(mse / var) if var > 0 else float("nan")


def one_round(ohe, feat, sample, train_idx, test_idx):
    return {
        "geometry_spearman": geometry_spearman(ohe[sample], feat[sample]),
        "ohe_to_feature_nmse": reconstruction_nmse(ohe, feat, train_idx, test_idx),
        "feature_to_ohe_nmse": reconstruction_nmse(feat, ohe, train_idx, test_idx),
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return np.mean(values), np.std(values)


def main():
    arrays = load_features(FEATURES_PT)
    ohe = arrays[ONEHOT_KEY]
    if N > len(ohe):
        raise ValueError(f"N={N} is larger than row count {len(ohe)}")
    if not 0 < TEST_FRAC < 1:
        raise ValueError("TEST_FRAC must be between 0 and 1")
    if N < 3:
        raise ValueError("N must be at least 3")

    rng = np.random.default_rng(SEED)
    feature_keys = [k for k in arrays if k != ONEHOT_KEY]
    if not feature_keys:
        raise ValueError(f"no feature keys besides {ONEHOT_KEY!r}")
    results = {k: [] for k in feature_keys}

    for i in range(1, K + 1):
        sample = rng.choice(len(ohe), size=N, replace=False)
        rng.shuffle(sample)
        cut = int(round(len(sample) * (1.0 - TEST_FRAC)))
        train_idx, test_idx = sample[:cut], sample[cut:]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("TEST_FRAC leaves an empty train or test split")

        for key in feature_keys:
            results[key].append(one_round(ohe, arrays[key], sample, train_idx, test_idx))
        if VERBOSE and (i == 1 or i % PRINT_EVERY == 0 or i == K):
            print(f"\nround {i}/{K}")
            print_summary(results)

    print("\nfinal")
    print_summary(results)


def print_summary(results):
    header = "feature  geom_rho  ohe->feat_nmse  feat->ohe_nmse"
    print(header)
    print("-" * len(header))
    for key, rows in results.items():
        geom = summarize([r["geometry_spearman"] for r in rows])
        o2f = summarize([r["ohe_to_feature_nmse"] for r in rows])
        f2o = summarize([r["feature_to_ohe_nmse"] for r in rows])
        print(
            f"{key:<8} "
            f"{geom[0]:>7.3f}+/-{geom[1]:<6.3f} "
            f"{o2f[0]:>7.3f}+/-{o2f[1]:<6.3f} "
            f"{f2o[0]:>7.3f}+/-{f2o[1]:<6.3f}"
        )


if __name__ == "__main__":
    main()
