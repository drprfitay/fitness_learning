#!/usr/bin/env python3
"""Create Hamming distance matrices or MDS figures from one-hot variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default=None, help="Dataset name under <root>/data/<dataset>/<dataset>.csv")
    parser.add_argument("--dataset_path", default=None, help="Explicit dataset CSV path")
    parser.add_argument("--root_path", default=str(DEFAULT_ROOT))
    parser.add_argument("--first_col", "--first_mutation_col", default=None)
    parser.add_argument("--last_col", "--last_mutation_col", default=None)
    parser.add_argument("--sequence_col", "--sequence_column_name", default="full_seq")
    parser.add_argument("--num_muts_col", "--num_muts_colname", default="num_muts")
    parser.add_argument("--activity_col", "--activity_column_name", default="activity")
    parser.add_argument("--N", type=int, default=None, help="Randomly sample N selected variants before adding WT")
    parser.add_argument("--nmuts", type=int, nargs="+", default=None, help="Keep only these num_muts values")
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument("--onehot_path", default=None, help="External OHE matrix: .npy, .npz, .pt, .csv, .tsv")
    parser.add_argument("--wt_onehot_path", default=None, help="External WT OHE vector path")
    parser.add_argument("--wt_index", type=int, default=None, help="Use this row from the external OHE matrix as WT")

    parser.add_argument("--output_type", choices=["npy", "figure", "both"], default="npy")
    parser.add_argument("--output_path", default=None, help="Output .npy path for distance matrix")
    parser.add_argument("--figure_path", default=None, help="Output figure path")
    parser.add_argument("--figure_format", choices=["png", "svg"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def dataset_csv_path(args) -> Path:
    if args.dataset_path:
        return Path(args.dataset_path).expanduser()
    if not args.dataset_name:
        raise ValueError("provide --dataset_name or --dataset_path")
    return Path(args.root_path).expanduser() / "data" / args.dataset_name / f"{args.dataset_name}.csv"


def mutation_columns(df: pd.DataFrame, first_col: str | None, last_col: str | None, sequence_col: str) -> list[str]:
    columns = list(df.columns)
    if first_col is not None or last_col is not None:
        if first_col not in columns or last_col not in columns:
            raise ValueError(f"could not find mutation columns {first_col!r} and {last_col!r}")
        si = columns.index(first_col)
        ei = columns.index(last_col) + 1
        if ei <= si:
            raise ValueError("--last_col must appear after --first_col")
        return columns[si:ei]

    stop_candidates = [col for col in [sequence_col, "sanity_mut_numb", "activity", "num_muts"] if col in columns]
    if not stop_candidates:
        raise ValueError("could not infer mutation columns; provide --first_col and --last_col")
    stop = min(columns.index(col) for col in stop_candidates)
    inferred = columns[:stop]
    if not inferred:
        raise ValueError("inferred zero mutation columns; provide --first_col and --last_col")
    print(f"[hamming_mds] inferred mutation columns: first={inferred[0]} last={inferred[-1]} count={len(inferred)}")
    return inferred


def select_dataframe(df: pd.DataFrame, args) -> tuple[pd.DataFrame, np.ndarray]:
    selected = df.copy()
    if args.nmuts is not None:
        if args.num_muts_col not in selected.columns:
            raise ValueError(f"--nmuts requires column {args.num_muts_col!r}")
        selected = selected[selected[args.num_muts_col].astype(int).isin(args.nmuts)]

    if args.N is not None:
        if args.N <= 0:
            raise ValueError("--N must be positive")
        if args.N < len(selected):
            selected = selected.sample(n=args.N, random_state=args.random_seed)

    indices = selected.index.to_numpy(dtype=int)
    return selected.reset_index(drop=True), indices


def build_dataset_onehot(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    csv_path = dataset_csv_path(args)
    df = pd.read_csv(csv_path)
    selected, selected_indices = select_dataframe(df, args)
    mut_cols = mutation_columns(df, args.first_col, args.last_col, args.sequence_col)

    if args.num_muts_col in df.columns and (df[args.num_muts_col] == 0).any():
        wt_row = df.loc[df[args.num_muts_col] == 0, mut_cols].iloc[[0]]
        if args.num_muts_col in selected.columns:
            keep = (selected[args.num_muts_col] != 0).to_numpy()
            selected = selected.loc[keep].reset_index(drop=True)
            selected_indices = selected_indices[keep]
    else:
        wt_values = {col: col[0] for col in mut_cols}
        wt_row = pd.DataFrame([wt_values])

    combined = pd.concat([wt_row[mut_cols], selected[mut_cols]], ignore_index=True)
    onehot = pd.get_dummies(combined, columns=mut_cols).astype(float).to_numpy()
    wt_ohe = onehot[0]
    x_ohe = onehot[1:]
    print(f"[hamming_mds] dataset={csv_path}")
    print(f"[hamming_mds] variants={x_ohe.shape[0]} ohe_features={x_ohe.shape[1]}")
    return x_ohe, wt_ohe, selected_indices, csv_path


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"{path} contains multiple arrays; save a single-array npz or use npy")
        return data[data.files[0]]
    if suffix == ".pt":
        import torch

        value = torch.load(path, map_location="cpu")
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    if suffix == ".csv":
        return pd.read_csv(path).to_numpy()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t").to_numpy()
    raise ValueError(f"unsupported array path suffix: {path}")


def build_external_onehot(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path | None]:
    if args.output_type in {"npy", "both"} and args.output_path is None:
        raise ValueError("--onehot_path mode requires --output_path when saving a matrix")
    if args.output_type in {"figure", "both"} and args.figure_path is None:
        raise ValueError("--onehot_path mode requires --figure_path when saving a figure")

    x_all = np.asarray(load_array(args.onehot_path), dtype=float)
    if x_all.ndim != 2:
        raise ValueError(f"external onehot matrix must be 2D, got shape {x_all.shape}")

    if args.dataset_path or args.dataset_name:
        csv_path = dataset_csv_path(args)
        df = pd.read_csv(csv_path)
        selected_df, selected_indices = select_dataframe(df, args)
        selected_indices = selected_indices.astype(int)
        if x_all.shape[0] != len(df):
            raise ValueError(f"external onehot rows {x_all.shape[0]} do not match dataset rows {len(df)}")
    else:
        csv_path = None
        selected_indices = np.arange(x_all.shape[0], dtype=int)
        if args.nmuts is not None:
            raise ValueError("--nmuts with --onehot_path requires --dataset_name or --dataset_path")
        if args.N is not None and args.N < len(selected_indices):
            rng = np.random.default_rng(args.random_seed)
            selected_indices = rng.choice(selected_indices, size=args.N, replace=False)

    if args.wt_onehot_path:
        wt_ohe = np.asarray(load_array(args.wt_onehot_path), dtype=float).reshape(-1)
    elif args.wt_index is not None:
        wt_ohe = x_all[int(args.wt_index)].reshape(-1)
        selected_indices = selected_indices[selected_indices != int(args.wt_index)]
    else:
        raise ValueError("--onehot_path requires --wt_onehot_path or --wt_index")

    x_ohe = x_all[selected_indices]
    if wt_ohe.shape[0] != x_ohe.shape[1]:
        raise ValueError(f"WT OHE length {wt_ohe.shape[0]} does not match matrix width {x_ohe.shape[1]}")

    print(f"[hamming_mds] external_onehot={args.onehot_path}")
    print(f"[hamming_mds] variants={x_ohe.shape[0]} ohe_features={x_ohe.shape[1]}")
    return x_ohe, wt_ohe, selected_indices, csv_path


def hamming_distance_matrix(x_ohe: np.ndarray, wt_ohe: np.ndarray) -> np.ndarray:
    x_ohe = np.asarray(x_ohe).astype(float)
    wt_ohe = np.asarray(wt_ohe).astype(float).reshape(1, -1)
    x_all = np.vstack([wt_ohe, x_ohe])
    return pairwise_distances(x_all, metric="manhattan") / 2.0


def default_output_path(args, csv_path: Path | None) -> Path:
    if args.output_path:
        output_path = Path(args.output_path).expanduser()
        if output_path.suffix.lower() != ".npy":
            return output_path / "hamming_distance.npy"
        return output_path
    if csv_path is None:
        raise ValueError("provide --output_path when no dataset path is available")
    return csv_path.parent / "hamming_distance.npy"


def default_figure_path(args, csv_path: Path | None) -> Path:
    if args.figure_path:
        figure_path = Path(args.figure_path).expanduser()
        if figure_path.suffix.lower() not in {".png", ".svg"}:
            return figure_path / f"hamming_mds.{args.figure_format}"
        return figure_path
    if args.output_path:
        output_path = Path(args.output_path).expanduser()
        if output_path.suffix.lower() != ".npy":
            return output_path / f"hamming_mds.{args.figure_format}"
    if csv_path is None:
        raise ValueError("provide --figure_path when no dataset path is available")
    return csv_path.parent / f"hamming_mds.{args.figure_format}"


def save_matrix(distance_matrix: np.ndarray, selected_indices: np.ndarray, args, csv_path: Path | None) -> None:
    output_path = default_output_path(args, csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, distance_matrix)
    np.save(output_path.with_suffix(".indices.npy"), selected_indices)
    print(f"[hamming_mds] saved distance matrix={output_path} shape={distance_matrix.shape}")
    print(f"[hamming_mds] saved selected indices={output_path.with_suffix('.indices.npy')}")


def save_figure(distance_matrix: np.ndarray, args, csv_path: Path | None) -> None:
    import matplotlib.pyplot as plt

    try:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=args.random_seed,
            normalized_stress="auto",
        )
    except TypeError:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=args.random_seed,
        )

    coords = mds.fit_transform(distance_matrix)
    coords = coords - coords[0]
    hamming_from_wt = distance_matrix[0, 1:]

    figure_path = default_figure_path(args, csv_path)
    figure_path = figure_path.with_suffix(f".{args.figure_format}")
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    sc = plt.scatter(coords[1:, 0], coords[1:, 1], c=hamming_from_wt, s=25, alpha=0.85)
    plt.scatter(coords[0, 0], coords[0, 1], s=160, marker="*", edgecolor="black", linewidth=1.2, label="WT")
    plt.colorbar(sc, label="Hamming distance from WT")
    plt.xlabel("Sequence space (MDS 1)")
    plt.ylabel("Sequence space (MDS 2)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=args.dpi if args.figure_format == "png" else None)
    plt.close()
    print(f"[hamming_mds] saved figure={figure_path}")


def main():
    args = parse_args()
    if args.onehot_path:
        x_ohe, wt_ohe, selected_indices, csv_path = build_external_onehot(args)
    else:
        x_ohe, wt_ohe, selected_indices, csv_path = build_dataset_onehot(args)

    distance_matrix = hamming_distance_matrix(x_ohe, wt_ohe)

    if args.output_type in {"npy", "both"}:
        save_matrix(distance_matrix, selected_indices, args, csv_path)
    if args.output_type in {"figure", "both"}:
        save_figure(distance_matrix, args, csv_path)


if __name__ == "__main__":
    main()
