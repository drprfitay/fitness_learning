#!/usr/bin/env python3
"""t-SNE contour plots for nominated sequence libraries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from utils_for_analysis import DATASET_PATHS, num_muts_column_name, positions


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "esm_8m"


def resolve_path(path):
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def default_dataset_path(dataset):
    if dataset not in DATASET_PATHS:
        raise ValueError("unknown dataset %r; pass --dataset_path or use one of: %s" % (dataset, ", ".join(sorted(DATASET_PATHS))))
    return resolve_path(DATASET_PATHS[dataset])


def default_embeddings_path(dataset, model):
    return SCRIPT_DIR / "data" / dataset / "embeddings" / model / "embeddings.pt"


def relevant_mutation_columns(dataset, df, first_col=None, last_col=None):
    if first_col is None or last_col is None:
        if dataset not in positions:
            raise ValueError("no default mutation-column range for %r; pass --first_col and --last_col" % dataset)
        first_col, last_col = positions[dataset]
    columns = np.asarray(df.columns)
    start = np.where(columns == first_col)[0]
    end = np.where(columns == last_col)[0]
    if len(start) == 0 or len(end) == 0:
        raise ValueError("could not find mutation columns %r and %r" % (first_col, last_col))
    si = int(start[0])
    ei = int(end[0]) + 1
    if ei <= si:
        raise ValueError("--last_col must appear after --first_col")
    return df.columns[si:ei]


def one_hot_for_libraries(df, dataset, first_col=None, last_col=None):
    relevant_columns = relevant_mutation_columns(dataset, df, first_col=first_col, last_col=last_col)
    one_hot = pd.get_dummies(df[relevant_columns])
    return one_hot.to_numpy(dtype=np.int8), one_hot.columns


def load_embeddings(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().float().numpy()
    else:
        arr = np.asarray(obj, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.nanmean(arr, axis=1)
    if arr.ndim != 2:
        raise ValueError("expected embeddings with shape (N, D) or (N, L, D), got %s" % (arr.shape,))
    return arr.astype(np.float32, copy=False)


def normalize_embeddings(embeddings):
    mu = np.nanmean(embeddings, axis=0)
    sigma = np.nanstd(embeddings, axis=0)
    sigma[sigma == 0] = 1.0
    normalized = (embeddings - mu) / sigma
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def top_variant_indices_from_feature_columns(one_hot, feature_indices, k):
    feature_indices = np.asarray(feature_indices, dtype=int)
    if np.any(feature_indices < 0) or np.any(feature_indices >= one_hot.shape[1]):
        raise ValueError("library feature indices must be between 0 and %d" % (one_hot.shape[1] - 1))
    scores = one_hot[:, feature_indices].sum(axis=1)
    return np.argsort(-scores)[: int(k)]


def infer_wt_idx(df, dataset, wt_idx=None, num_muts_column=None):
    if wt_idx is not None:
        wt_idx = int(wt_idx)
        if wt_idx < 0 or wt_idx >= len(df):
            raise ValueError("--wt_idx must be between 0 and %d" % (len(df) - 1))
        return wt_idx

    if num_muts_column is None:
        num_muts_column = num_muts_column_name.get(dataset)
    if num_muts_column is None or num_muts_column not in df.columns:
        raise ValueError("could not infer WT row; pass --wt_idx or --num_muts_column")

    zero_mut_rows = np.flatnonzero(df[num_muts_column].astype(int).to_numpy() == 0)
    if len(zero_mut_rows) == 0:
        raise ValueError("no %s == 0 row found; pass --wt_idx for mutation_columns mode" % num_muts_column)
    return int(zero_mut_rows[0])


def real_mutation_columns_from_wt(one_hot, wt_idx):
    return np.flatnonzero(one_hot[int(wt_idx), :] == 0).astype(int)


def get_candidate_library_fast(ohe, chosen_mutations, total_mutations, max_outside_muts=1):
    chosen_mutations = np.asarray(chosen_mutations, dtype=int)
    inside = ohe[:, chosen_mutations].sum(axis=1)
    outside = total_mutations - inside
    candidate_mask = (inside > 0) & (outside <= int(max_outside_muts))
    return np.flatnonzero(candidate_mask)


def sample_candidate_library(name, candidate_pool, k, rng):
    candidate_pool = np.asarray(candidate_pool, dtype=int)
    if len(candidate_pool) == 0:
        raise ValueError("%s nominated mutations produced an empty candidate library" % name)
    n = min(int(k), len(candidate_pool))
    if n < int(k):
        print("[WARNING] %s candidate library has only %d variants; plotting all of them" % (name, len(candidate_pool)))
    return rng.choice(candidate_pool, size=n, replace=False)


def candidate_indices(
    name,
    provided_indices,
    one_hot,
    k,
    index_mode,
    n_rows,
    rng,
    real_mutation_cols=None,
    total_mutations=None,
    max_outside_muts=1,
):
    provided_indices = np.asarray(provided_indices, dtype=int)
    if index_mode == "feature_columns":
        return top_variant_indices_from_feature_columns(one_hot, provided_indices, k)
    if index_mode == "mutation_columns":
        if real_mutation_cols is None or total_mutations is None:
            raise ValueError("mutation_columns mode requires real_mutation_cols and total_mutations")
        if np.any(provided_indices < 0) or np.any(provided_indices >= one_hot.shape[1]):
            raise ValueError("%s mutation column indices must be between 0 and %d" % (name, one_hot.shape[1] - 1))
        non_mutation_cols = np.setdiff1d(provided_indices, real_mutation_cols, assume_unique=False)
        if len(non_mutation_cols) > 0:
            raise ValueError(
                "%s includes OHE columns that are not real mutations relative to the WT row: %s" %
                (name, " ".join(map(str, non_mutation_cols[:20])))
            )
        candidate_pool = get_candidate_library_fast(
            one_hot,
            chosen_mutations=provided_indices,
            total_mutations=total_mutations,
            max_outside_muts=max_outside_muts,
        )
        print("%s candidate library size before K sampling: %d" % (name, len(candidate_pool)))
        return sample_candidate_library(name, candidate_pool, k, rng)
    if np.any(provided_indices < 0) or np.any(provided_indices >= n_rows):
        raise ValueError("%s variant-row indices must be between 0 and %d" % (name, n_rows - 1))
    return provided_indices[: int(k)]


def sample_background_indices(n_rows, selected_indices, sample_size, seed):
    rng = np.random.default_rng(seed)
    selected = np.unique(np.asarray(selected_indices, dtype=int))
    all_indices = np.arange(n_rows)
    remaining = np.setdiff1d(all_indices, selected, assume_unique=False)
    n_sample = min(int(sample_size), len(remaining))
    sampled = rng.choice(remaining, size=n_sample, replace=False) if n_sample > 0 else np.asarray([], dtype=int)
    return np.unique(np.concatenate([sampled, selected])).astype(int)


def run_tsne(embeddings, perplexity, learning_rate, seed):
    n = embeddings.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points for t-SNE")
    perplexity = min(float(perplexity), max(1.0, (n - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=seed, init="pca")
    return tsne.fit_transform(embeddings)


def contour_tsne_values(
    ax,
    tsne,
    z,
    selected_local_indices,
    bandwidth=1.0,
    k=50,
    support_quantile=99,
    levels=25,
    pad=0.5,
    grid_jump=0.25,
):
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)
    if finite.sum() < 3:
        raise ValueError("need at least 3 finite activity values for contour plot")

    X = tsne[finite]
    z_fit = z[finite]
    x = X[:, 0]
    y = X[:, 1]

    x_grid = np.arange(x.min() - pad, x.max() + pad + grid_jump, grid_jump)
    y_grid = np.arange(y.min() - pad, y.max() + pad + grid_jump, grid_jump)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    k = min(int(k), len(X))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    train_kdist = nbrs.kneighbors(X)[0][:, -1]
    threshold = np.percentile(train_kdist, support_quantile)
    grid_kdist = nbrs.kneighbors(grid)[0][:, -1]
    support_mask = grid_kdist <= threshold

    zhat = np.full(len(grid), np.nan)
    grid_in = grid[support_mask]
    dx = grid_in[:, None, 0] - X[None, :, 0]
    dy = grid_in[:, None, 1] - X[None, :, 1]
    dist2 = dx**2 + dy**2
    weights = np.exp(-dist2 / (2 * bandwidth**2))
    zhat[support_mask] = (weights @ z_fit) / (weights.sum(axis=1) + 1e-12)
    zgrid = zhat.reshape(xx.shape)

    ax.contourf(xx, yy, zgrid, levels=levels, alpha=0.75, cmap="OrRd")
    ax.contour(xx, yy, zgrid, levels=levels, colors="black", linewidths=0.4, alpha=0.5)

    selected_local_indices = np.asarray(selected_local_indices, dtype=int)
    if len(selected_local_indices) > 0:
        ax.scatter(
            tsne[selected_local_indices, 0],
            tsne[selected_local_indices, 1],
            c="black",
            s=40,
            edgecolor="white",
            marker="X",
            linewidth=0.6,
            zorder=10,
        )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_libraries(tsne_coords, z, sample_indices, library_candidates, output_path, args):
    local_lookup = {int(global_idx): i for i, global_idx in enumerate(sample_indices)}
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    panel_order = [("best", "Best"), ("worst", "Worst"), ("pssm", "PSSM"), ("plm", "PLM")]
    for ax, (key, title) in zip(axes, panel_order):
        selected_local = [local_lookup[int(idx)] for idx in library_candidates[key] if int(idx) in local_lookup]
        contour_tsne_values(
            ax,
            tsne_coords,
            z,
            selected_local,
            bandwidth=args.bandwidth,
            k=args.contour_k,
            support_quantile=args.support_quantile,
            levels=args.levels,
            pad=args.pad,
            grid_jump=args.grid_jump,
        )
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=output_path.suffix.lstrip(".") or "svg")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embeddings_path", default=None)
    parser.add_argument("--activity_column", default="activity")
    parser.add_argument("--first_col", default=None)
    parser.add_argument("--last_col", default=None)
    parser.add_argument("--S", type=int, default=None, help="Library S size, recorded in the default output name.")
    parser.add_argument("--K", type=int, default=20, help="Number of selected candidate variants per library.")
    parser.add_argument("--best_indices", type=int, nargs="+", required=True)
    parser.add_argument("--worst_indices", type=int, nargs="+", required=True)
    parser.add_argument("--pssm_indices", type=int, nargs="+", required=True)
    parser.add_argument("--plm_indices", type=int, nargs="+", required=True)
    parser.add_argument("--library_index_mode", choices=["feature_columns", "mutation_columns", "variant_rows"], default="feature_columns")
    parser.add_argument("--max_outside_muts", type=int, default=1)
    parser.add_argument("--wt_idx", type=int, default=None)
    parser.add_argument("--num_muts_column", default=None)
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=60)
    parser.add_argument("--learning_rate", type=float, default=50)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--contour_k", type=int, default=50)
    parser.add_argument("--support_quantile", type=float, default=99)
    parser.add_argument("--levels", type=int, default=25)
    parser.add_argument("--pad", type=float, default=0.5)
    parser.add_argument("--grid_jump", type=float, default=0.25)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--tsne_output_path", default=None)
    parser.add_argument("--selected_output_path", default=None)
    return parser.parse_args()


def default_output_path(args):
    s_part = "S%s" % args.S if args.S is not None else "SNA"
    name = "tsne_contour_%s_%s_%s_K%d.svg" % (args.dataset, args.model, s_part, args.K)
    return SCRIPT_DIR / "data" / "tsne_embeddings" / name


def main():
    args = parse_args()
    dataset_path = resolve_path(args.dataset_path) if args.dataset_path else default_dataset_path(args.dataset)
    embeddings_path = resolve_path(args.embeddings_path) if args.embeddings_path else default_embeddings_path(args.dataset, args.model)
    output_path = resolve_path(args.output_path) if args.output_path else default_output_path(args)

    print("Dataset: %s" % args.dataset)
    print("Dataset path: %s" % dataset_path)
    print("Embedding model: %s" % args.model)
    print("Embeddings path: %s" % embeddings_path)
    print("Sample size: %d background + selected libraries" % args.sample_size)

    df = pd.read_csv(dataset_path)
    if args.activity_column not in df.columns:
        raise ValueError("activity column %r not found in %s" % (args.activity_column, dataset_path))
    activity = df[args.activity_column].to_numpy(dtype=float)

    embeddings = load_embeddings(embeddings_path)
    if embeddings.shape[0] != len(df):
        raise ValueError("embedding rows (%d) do not match dataset rows (%d)" % (embeddings.shape[0], len(df)))

    one_hot, one_hot_columns = one_hot_for_libraries(df, args.dataset, first_col=args.first_col, last_col=args.last_col)
    rng = np.random.default_rng(args.seed)
    real_mutation_cols = None
    total_mutations = None
    if args.library_index_mode == "mutation_columns":
        wt_idx = infer_wt_idx(df, args.dataset, wt_idx=args.wt_idx, num_muts_column=args.num_muts_column)
        real_mutation_cols = real_mutation_columns_from_wt(one_hot, wt_idx)
        total_mutations = one_hot[:, real_mutation_cols].sum(axis=1)
        print("WT row index: %d" % wt_idx)
        print("Real mutation OHE columns: %d / %d" % (len(real_mutation_cols), one_hot.shape[1]))
        print("max_outside_muts: %d" % args.max_outside_muts)

    libraries = {
        "best": candidate_indices(
            "best",
            args.best_indices,
            one_hot,
            args.K,
            args.library_index_mode,
            len(df),
            rng,
            real_mutation_cols=real_mutation_cols,
            total_mutations=total_mutations,
            max_outside_muts=args.max_outside_muts,
        ),
        "worst": candidate_indices(
            "worst",
            args.worst_indices,
            one_hot,
            args.K,
            args.library_index_mode,
            len(df),
            rng,
            real_mutation_cols=real_mutation_cols,
            total_mutations=total_mutations,
            max_outside_muts=args.max_outside_muts,
        ),
        "pssm": candidate_indices(
            "pssm",
            args.pssm_indices,
            one_hot,
            args.K,
            args.library_index_mode,
            len(df),
            rng,
            real_mutation_cols=real_mutation_cols,
            total_mutations=total_mutations,
            max_outside_muts=args.max_outside_muts,
        ),
        "plm": candidate_indices(
            "plm",
            args.plm_indices,
            one_hot,
            args.K,
            args.library_index_mode,
            len(df),
            rng,
            real_mutation_cols=real_mutation_cols,
            total_mutations=total_mutations,
            max_outside_muts=args.max_outside_muts,
        ),
    }
    selected_all = np.unique(np.concatenate(list(libraries.values()))).astype(int)
    sample_indices = sample_background_indices(len(df), selected_all, args.sample_size, args.seed)
    sample_indices.sort()

    print("Selected candidate rows: %d unique" % len(selected_all))
    print("Total t-SNE rows: %d" % len(sample_indices))
    for name, idx in libraries.items():
        print("%s candidates: %s" % (name, " ".join(map(str, idx[: min(len(idx), 20)]))))

    X = normalize_embeddings(embeddings[sample_indices])
    tsne_coords = run_tsne(X, args.perplexity, args.learning_rate, args.seed)
    z = activity[sample_indices]

    if args.tsne_output_path:
        tsne_output_path = resolve_path(args.tsne_output_path)
        tsne_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tsne_output_path, {"indices": sample_indices, "tsne": tsne_coords})
        print("Saved t-SNE coordinates to %s" % tsne_output_path)

    selected_output_path = resolve_path(args.selected_output_path) if args.selected_output_path else output_path.with_suffix(".selected.csv")
    selected_rows = []
    for library_name, indices in libraries.items():
        for rank, row_idx in enumerate(indices, start=1):
            selected_rows.append({
                "library": library_name,
                "rank": rank,
                "row_index": int(row_idx),
                "activity": activity[int(row_idx)],
                "library_index_mode": args.library_index_mode,
                "max_outside_muts": args.max_outside_muts if args.library_index_mode == "mutation_columns" else np.nan,
            })
    pd.DataFrame(selected_rows).to_csv(selected_output_path, index=False)
    print("Saved selected candidates to %s" % selected_output_path)

    plot_libraries(tsne_coords, z, sample_indices, libraries, output_path, args)
    print("Saved plot to %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
