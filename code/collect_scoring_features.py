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
    parser.add_argument("--embedding_dir", "--embeddings_dir", required=True, nargs="+")
    parser.add_argument("--sequence_df_path", "--sequence_path", required=True)
    parser.add_argument("--first_col", "--first_mutation_col", required=True)
    parser.add_argument("--last_col", "--last_mutation_col", required=True)
    parser.add_argument("--activity_col", "--activity_column_name", default="activity")
    parser.add_argument("--num_muts_col", "--num_muts_colname", default="num_muts")
    parser.add_argument("--max_num_muts", type=int, default=None, help="Only collect rows with num_muts <= this value.")
    parser.add_argument("--output", default=None, help="Output .pt path. Defaults to <sequence_df_dir>/features.pt")
    parser.add_argument("--embedding_key", default=None, nargs="+", help="Feature key for embeddings. Defaults to each embedding directory name.")
    parser.add_argument("--onehot_key", default="onehot")
    parser.add_argument("--positions_to_select", type=int, nargs="+", default=None, help="Positions to select when saved embeddings contain the full sequence.")
    parser.add_argument("--position_indexing", choices=["model", "embedding"], default="model", help="'model' uses one-based model positions. 'embedding' uses zero-based embedding tensor indices.")
    parser.add_argument("--position_collection", choices=["as_saved", "full", "partial", "both"], default=["as_saved"], nargs="+", help="Which position views to store per embedding dir. Use 'both' for full_<key> and partial_<key>.")
    parser.add_argument("--sequence_colname", "--sequence_col", default=None, help="Sequence column used to infer full sequence length.")
    parser.add_argument("--full_sequence_length", type=int, default=None, help="Explicit full sequence length for deciding if saved embeddings contain all positions.")
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


def infer_full_sequence_length(df, args):
    if args.full_sequence_length is not None:
        return int(args.full_sequence_length)

    candidate_cols = []
    if args.sequence_colname is not None:
        candidate_cols.append(args.sequence_colname)
    candidate_cols.extend(["full_seq", "full_sequence", "seq", "sequence"])

    for col in candidate_cols:
        if col not in df.columns:
            continue
        lengths = df[col].dropna().astype(str).str.len().unique()
        if len(lengths) == 0:
            continue
        if len(lengths) > 1:
            print(f"[collect] sequence lengths in {col!r} are not all identical; using first observed length {int(lengths[0])}")
        return int(lengths[0])

    return None


def select_embedding_positions(emb, positions_to_select, position_indexing, full_sequence_length, position_collection, embedding_key, nmut):
    if position_collection == "full":
        if full_sequence_length is not None and emb.ndim > 2 and emb.shape[1] != full_sequence_length:
            print(
                f"[collect] {embedding_key} nmut={nmut}: requested full positions, "
                f"but saved embeddings have {emb.shape[1]} positions; expected full length {full_sequence_length}. Keeping saved positions."
            )
        return emb

    if position_collection != "partial":
        return emb

    requested = 0 if positions_to_select is None else len(positions_to_select)
    if positions_to_select is None:
        print(f"[collect] {embedding_key} nmut={nmut}: requested partial positions but no --positions_to_select was provided; keeping saved positions.")
        return emb
    if emb.ndim <= 2:
        print(f"[collect] {embedding_key} nmut={nmut}: embeddings are already 2D; cannot select {requested} positions. Keeping saved features.")
        return emb
    if full_sequence_length is None:
        print(f"[collect] {embedding_key} nmut={nmut}: cannot infer full sequence length; cannot validate/select {requested} positions. Keeping saved {emb.shape[1]} positions.")
        return emb
    if emb.shape[1] != full_sequence_length:
        print(
            f"[collect] {embedding_key} nmut={nmut}: requested {requested} positions, "
            f"but saved embeddings have {emb.shape[1]} positions and expected full length is {full_sequence_length}. "
            "Assuming embeddings are already preselected and keeping saved positions."
        )
        return emb

    if position_indexing == "model":
        bad_positions = [pos for pos in positions_to_select if pos < 1]
        if bad_positions:
            raise ValueError(f"model positions must be >= 1: {bad_positions}")
        position_indices = [pos - 1 for pos in positions_to_select]
    else:
        bad_positions = [pos for pos in positions_to_select if pos < 0]
        if bad_positions:
            raise ValueError(f"embedding positions must be >= 0: {bad_positions}")
        position_indices = positions_to_select

    if max(position_indices) >= emb.shape[1]:
        raise IndexError(f"requested position index {max(position_indices)} but embeddings only have {emb.shape[1]} positions")

    print(f"[collect] {embedding_key} nmut={nmut}: selecting {requested} positions from full saved embeddings with {emb.shape[1]} positions.")
    return emb[:, position_indices, :]


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


def selected_dataframe_indices(df, args):
    if args.max_num_muts is None:
        return np.arange(len(df), dtype=int)
    if args.num_muts_col not in df.columns:
        raise KeyError(f"num_muts column {args.num_muts_col!r} was not found")
    selected = np.where(df[args.num_muts_col].astype(int).to_numpy() <= args.max_num_muts)[0]
    print(f"[collect] selected {len(selected)}/{len(df)} rows with {args.num_muts_col} <= {args.max_num_muts}")
    return selected.astype(int)


def load_embedding_chunks(args, df, selected_indices, position_collection, embedding_key, full_sequence_length):
    embedding_dir = Path(args.embedding_dir).expanduser()
    n_rows = len(df)
    chunks = []
    index_chunks = []
    selected_set = set(np.asarray(selected_indices, dtype=int).tolist())
    output_row_by_df_index = {int(idx): pos for pos, idx in enumerate(selected_indices)}

    for k in discover_nmuts(embedding_dir):
        if args.max_num_muts is not None and k > args.max_num_muts:
            continue
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

        keep = np.array([int(row_idx) in selected_set for row_idx in idx], dtype=bool)
        if not np.any(keep):
            continue
        emb = emb[keep]
        idx = idx[keep]

        emb = select_embedding_positions(
            emb,
            args.positions_to_select,
            args.position_indexing,
            full_sequence_length,
            position_collection,
            embedding_key,
            k,
        )
        emb = prepare_embedding_array(emb, mean_embeddings=args.mean)
        chunks.append(emb)
        index_chunks.append(np.array([output_row_by_df_index[int(row_idx)] for row_idx in idx], dtype=int))
        print(f"[collect] nmut={k} embeddings={emb.shape} indices={idx.shape}")

    if not chunks:
        raise FileNotFoundError(f"no embedding chunks were loaded from {embedding_dir}")

    dim = chunks[0].shape[1]
    out = np.full((len(selected_indices), dim), np.nan, dtype=np.float32)
    seen = np.zeros(len(selected_indices), dtype=bool)
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
        missing_df_rows = np.asarray(selected_indices)[missing[:10]].tolist()
        raise AssertionError(f"missing embeddings for {len(missing)} selected dataframe rows; first original rows: {missing_df_rows}")
    return out


def expand_per_embedding_arg(values, n, name):
    if values is None:
        return None
    if len(values) == 1:
        return list(values) * n
    if len(values) != n:
        raise ValueError(f"--{name} must have either one value or the same number of values as --embedding_dir")
    return list(values)


def embedding_feature_specs(embedding_dirs, embedding_keys, position_collections):
    specs = []
    for embedding_dir, embedding_key, position_collection in zip(embedding_dirs, embedding_keys, position_collections):
        modes = ["full", "partial"] if position_collection == "both" else [position_collection]
        for mode in modes:
            feature_key = embedding_key if mode == "as_saved" else f"{mode}_{embedding_key}"
            specs.append((embedding_dir, feature_key, mode))
    keys = [spec[1] for spec in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("expanded embedding feature keys must be unique")
    return specs


def main():
    args = parse_args()
    if args.positions_to_select is not None and args.position_collection == ["as_saved"]:
        args.position_collection = ["partial"]
        print("[collect] --positions_to_select was provided; using --position_collection partial")

    sequence_df_path = Path(args.sequence_df_path).expanduser()
    df = pd.read_csv(sequence_df_path)
    if args.activity_col not in df.columns:
        raise KeyError(f"activity column {args.activity_col!r} was not found in {sequence_df_path}")
    selected_indices = selected_dataframe_indices(df, args)
    selected_df = df.iloc[selected_indices]
    full_sequence_length = infer_full_sequence_length(df, args)
    if full_sequence_length is None:
        print("[collect] could not infer full sequence length; position selection will keep saved positions.")
    else:
        print(f"[collect] inferred full sequence length: {full_sequence_length}")

    embedding_dirs = args.embedding_dir
    embedding_keys = args.embedding_key
    if embedding_keys is None:
        embedding_keys = [Path(embedding_dir).expanduser().name for embedding_dir in embedding_dirs]
    position_collections = expand_per_embedding_arg(args.position_collection, len(embedding_dirs), "position_collection")
    if len(embedding_keys) != len(embedding_dirs):
        raise ValueError("--embedding_key must have the same number of values as --embedding_dir")
    if len(set(embedding_keys)) != len(embedding_keys):
        raise ValueError("--embedding_key values must be unique")
    embedding_specs = embedding_feature_specs(embedding_dirs, embedding_keys, position_collections)

    mutation_cols = relevant_mutation_columns(df, args.first_col, args.last_col)
    onehot = pd.get_dummies(selected_df[mutation_cols]).astype(np.float32).to_numpy()
    activity = selected_df[args.activity_col].astype(float).to_numpy()

    output = Path(args.output).expanduser() if args.output else sequence_df_path.parent / "features.pt"
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    payload = {
        str(args.activity_col): torch.as_tensor(activity),
        str(args.onehot_key): torch.as_tensor(onehot),
    }

    for embedding_dir, embedding_key, position_collection in embedding_specs:
        args.embedding_dir = embedding_dir
        embeddings = load_embedding_chunks(args, df, selected_indices, position_collection, embedding_key, full_sequence_length)
        payload[str(embedding_key)] = torch.as_tensor(embeddings)
        print(f"[collect] {embedding_key}={embeddings.shape}")

    torch.save(payload, output)
    print(f"[collect] onehot={onehot.shape} activity={activity.shape}")
    print(f"[collect] wrote {output}")


if __name__ == "__main__":
    main()
