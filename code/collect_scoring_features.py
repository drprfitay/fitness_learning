#!/usr/bin/env python3
"""Collect chunked embeddings, one-hot mutations, and activity into features.pt."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


EMBEDDING_PREFIXES = (
    "embeddings_of_nmut",
    "embeddings_of_nmuts",
    "embedding_of_nmut",
    "embedding_of_nmuts",
    "embeddings_nmut",
    "embeddings_nmuts",
    "embedding_nmut",
    "embedding_nmuts",
)
INDEX_PREFIXES = (
    "indices_of_nmut",
    "indices_of_nmuts",
    "index_of_nmut",
    "index_of_nmuts",
    "indices_nmut",
    "indices_nmuts",
    "index_nmut",
    "index_nmuts",
)
Y_PREFIXES = (
    "y_values_of_nmut",
    "y_values_of_nmuts",
    "y_value_of_nmut",
    "y_value_of_nmuts",
    "y_values_nmut",
    "y_values_nmuts",
    "y_activity_of_nmut",
    "y_activity_of_nmuts",
    "y_activity_nmut",
    "y_activity_nmuts",
    "activity_of_nmut",
    "activity_of_nmuts",
    "activity_nmut",
    "activity_nmuts",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding_dir", "--embeddings_dir", required=True)
    parser.add_argument("--sequence_df_path", "--sequence_path", required=True)
    parser.add_argument("--first_col", "--first_mutation_col", required=True)
    parser.add_argument("--last_col", "--last_mutation_col", required=True)
    parser.add_argument("--activity_col", "--activity_column_name", default="activity")
    parser.add_argument("--num_muts_col", "--num_muts_colname", default="num_muts")
    parser.add_argument("--output", default=None, help="Output .pt path. Defaults to <sequence_df_dir>/features.pt")
    parser.add_argument("--embedding_key", default=None, help="Feature key for embeddings. Defaults to the embedding directory name.")
    parser.add_argument("--onehot_key", default="onehot")
    parser.add_argument("--mean", dest="mean", action="store_true", help="Average sequence/position embeddings to [N, D].")
    parser.add_argument("--flat", dest="mean", action="store_false", help="Flatten embeddings to [N, ...].")
    parser.set_defaults(mean=True)
    parser.add_argument("--skip_y_chunk_check", action="store_true")
    return parser.parse_args()


def tensor_to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def torch_load(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_embedding_array(emb, mean_embeddings):
    emb = np.asarray(emb)
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
    elif emb.ndim > 2:
        emb = emb.mean(axis=1) if mean_embeddings else emb.reshape(emb.shape[0], -1)
    return emb.astype(np.float32)


def discover_nmuts(embedding_dir):
    values = []
    for path in Path(embedding_dir).glob("*.pt"):
        stem = path.stem
        if "embedding" not in stem or "nmut" not in stem:
            continue
        match = re.search(r"(?:^|_)(\d+)$", stem)
        if match:
            values.append(int(match.group(1)))
    values = sorted(set(values))
    if not values:
        raise FileNotFoundError(f"no embedding chunk files found in {embedding_dir}")
    return values


def find_numbered_file(directory, prefixes, k, contains):
    directory = Path(directory)
    for prefix in prefixes:
        for name in (f"{prefix}_{k}.pt", f"{prefix}{k}.pt"):
            path = directory / name
            if path.is_file():
                return path

    candidates = []
    for path in directory.glob("*.pt"):
        stem = path.stem
        if contains not in stem or "nmut" not in stem:
            continue
        match = re.search(r"(?:^|_)(\d+)$", stem)
        if match and int(match.group(1)) == int(k):
            candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise FileNotFoundError(f"ambiguous {contains} chunk for nmut={k}: {[str(p) for p in candidates]}")
    tried = ", ".join(f"{prefix}_{k}.pt" for prefix in prefixes)
    raise FileNotFoundError(f"missing {contains} chunk for nmut={k} in {directory}; tried {tried}")


def relevant_mutation_columns(df, first_col, last_col):
    columns = np.asarray(df.columns)
    start = np.where(columns == first_col)[0]
    end = np.where(columns == last_col)[0]
    if len(start) == 0 or len(end) == 0:
        raise ValueError(f"could not find mutation columns {first_col!r} and {last_col!r}")
    si = int(start[0])
    ei = int(end[0]) + 1
    if ei <= si:
        raise ValueError("--last_col must appear after --first_col")
    return df.columns[si:ei]


def load_embedding_chunks(args, df):
    embedding_dir = Path(args.embedding_dir).expanduser()
    n_rows = len(df)
    chunks = []
    index_chunks = []

    for k in discover_nmuts(embedding_dir):
        emb_path = find_numbered_file(embedding_dir, EMBEDDING_PREFIXES, k, "embedding")
        idx_path = find_numbered_file(embedding_dir, INDEX_PREFIXES, k, "indices")
        y_path = find_numbered_file(embedding_dir, Y_PREFIXES, k, "y")

        emb = tensor_to_numpy(torch_load(emb_path))
        idx = tensor_to_numpy(torch_load(idx_path)).astype(int).reshape(-1)
        y_values = tensor_to_numpy(torch_load(y_path)).reshape(-1)
        if emb.shape[0] != len(idx) or len(idx) != len(y_values):
            raise AssertionError(f"embedding/index/y length mismatch for nmut={k}")
        if np.any(idx < 0) or np.any(idx >= n_rows):
            raise AssertionError(f"indices for nmut={k} point outside dataframe rows")

        if args.num_muts_col in df.columns:
            df_nmuts = df.iloc[idx][args.num_muts_col].astype(int).to_numpy()
            if not np.all(df_nmuts == k):
                bad = idx[df_nmuts != k][:10]
                raise AssertionError(f"indices for nmut={k} point to rows with other nmuts: {bad.tolist()}")

        if not args.skip_y_chunk_check:
            df_y = df.iloc[idx][args.activity_col].astype(float).to_numpy()
            if not np.allclose(df_y, y_values.astype(float), equal_nan=True, rtol=1e-4, atol=1e-6):
                raise AssertionError(f"y chunk for nmut={k} does not match dataframe column {args.activity_col!r}")

        emb = prepare_embedding_array(emb, mean_embeddings=args.mean)
        chunks.append(emb)
        index_chunks.append(idx)
        print(f"[collect] nmut={k} embeddings={emb.shape} indices={idx.shape}")

    dim = chunks[0].shape[1]
    out = np.full((n_rows, dim), np.nan, dtype=np.float32)
    seen = np.zeros(n_rows, dtype=bool)
    for emb, idx in zip(chunks, index_chunks):
        if emb.shape[1] != dim:
            raise ValueError("embedding chunks have inconsistent feature dimensions")
        if np.any(seen[idx]):
            duplicated = idx[seen[idx]][:10]
            raise AssertionError(f"duplicate embedding rows detected: {duplicated.tolist()}")
        out[idx] = emb
        seen[idx] = True

    missing = np.where(~seen)[0]
    if len(missing):
        raise AssertionError(f"missing embeddings for {len(missing)} dataframe rows; first rows: {missing[:10].tolist()}")
    return out


def main():
    args = parse_args()
    sequence_df_path = Path(args.sequence_df_path).expanduser()
    df = pd.read_csv(sequence_df_path)
    if args.activity_col not in df.columns:
        raise KeyError(f"activity column {args.activity_col!r} was not found in {sequence_df_path}")

    mutation_cols = relevant_mutation_columns(df, args.first_col, args.last_col)
    onehot = pd.get_dummies(df[mutation_cols]).astype(np.float32).to_numpy()
    embeddings = load_embedding_chunks(args, df)
    activity = df[args.activity_col].astype(float).to_numpy()

    embedding_key = args.embedding_key or Path(args.embedding_dir).expanduser().name
    output = Path(args.output).expanduser() if args.output else sequence_df_path.parent / "features.pt"
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    payload = {
        str(args.activity_col): torch.as_tensor(activity),
        str(args.onehot_key): torch.as_tensor(onehot),
        str(embedding_key): torch.as_tensor(embeddings),
    }
    torch.save(payload, output)
    print(f"[collect] onehot={onehot.shape} embedding={embeddings.shape} activity={activity.shape}")
    print(f"[collect] wrote {output}")


if __name__ == "__main__":
    main()
