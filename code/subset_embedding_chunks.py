#!/usr/bin/env python3
"""Subset chunked embedding folders using original row indices.

Given a parent dataset directory such as ``his/`` with ``his.csv`` and
``embeddings/<model>/*_of_nmut_K.pt``, and subset directories such as ``his2/``
with ``his2.csv`` plus ``original_ind.pkl``, this script writes matching
chunked embedding folders under each subset directory.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


EMBEDDING_PREFIXES = ("embeddings_of_nmut", "embeddings_of_nmuts")
INDEX_PREFIXES = ("indices_of_nmut", "indices_of_nmuts")
Y_PREFIXES = ("y_values_of_nmut", "y_values_of_nmuts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_dir", required=True, help="Parent dataset directory, e.g. /path/to/his")
    parser.add_argument("--subset_dirs", nargs="+", required=True, help="Subset directories, e.g. /path/to/his2 /path/to/his5")
    parser.add_argument("--parent_csv", default=None, help="Defaults to <parent_dir>/<parent_dir_name>.csv")
    parser.add_argument("--subset_csv", default=None, help="Optional subset CSV name/path used for every subset directory.")
    parser.add_argument("--original_indices_name", default="original_ind.pkl")
    parser.add_argument("--parent_embeddings_dir", default=None, help="Defaults to <parent_dir>/embeddings")
    parser.add_argument("--output_embeddings_name", default="embeddings")
    parser.add_argument("--embedding_names", nargs="*", default=None, help="Optional embedding model folder names to process.")
    parser.add_argument("--parent_tokenized_cache_dir", default=None, help="Defaults to <parent_dir>/<parent_name>_cache or sibling <parent_name>_cache.")
    parser.add_argument("--tokenized_misc_name", default="misc")
    parser.add_argument("--copy_tokenized", action="store_true", help="Also subset pre-tokenized cache .pt files.")
    parser.add_argument("--tokenized_only", action="store_true", help="Only subset pre-tokenized cache .pt files.")
    parser.add_argument("--embeddings_only", action="store_true", help="Only subset embedding chunks.")
    parser.add_argument("--sequence_col", default="full_seq")
    parser.add_argument("--activity_col", default="activity")
    parser.add_argument("--num_muts_col", default="num_muts")
    parser.add_argument("--first_col_his", "--parent_first_col", default=None)
    parser.add_argument("--last_col_his", "--parent_last_col", default=None)
    parser.add_argument("--first_col_his2", default=None)
    parser.add_argument("--last_col_his2", default=None)
    parser.add_argument("--first_col_his5", default=None)
    parser.add_argument("--last_col_his5", default=None)
    parser.add_argument(
        "--subset_position_range",
        action="append",
        default=[],
        help="Optional generic subset range as subset_name:first_col:last_col.",
    )
    parser.add_argument("--min_nmut", "--min_muts", type=int, default=None)
    parser.add_argument("--max_nmut", "--max_muts", type=int, default=None)
    parser.add_argument("--write_aggregated", dest="write_aggregated", action="store_true")
    parser.add_argument("--no_write_aggregated", dest="write_aggregated", action="store_false")
    parser.add_argument("--validate_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.set_defaults(write_aggregated=True)
    return parser.parse_args()


def torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(value) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def to_cpu_tensor(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def row_count(value) -> int | None:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return int(value.shape[0]) if value.ndim > 0 else None
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return int(value.shape[0]) if value.ndim > 0 else None
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        counts = [row_count(item) for item in value.values()]
        counts = [count for count in counts if count is not None]
        if not counts:
            return None
        if len(set(counts)) != 1:
            return None
        return int(counts[0])
    return None


def subset_row_like(value, indices: np.ndarray, expected_parent_rows: int, *, label: str):
    count = row_count(value)
    if count is not None:
        if count != int(expected_parent_rows):
            raise AssertionError(f"{label} has {count} rows/items, expected parent rows={expected_parent_rows}")
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return value.detach().cpu()[torch.as_tensor(indices, dtype=torch.long)]
        except ImportError:
            pass
        if isinstance(value, np.ndarray):
            return value[indices]
        if isinstance(value, list):
            return [value[int(index)] for index in indices]
        if isinstance(value, tuple):
            return tuple(value[int(index)] for index in indices)

    if isinstance(value, dict):
        subset = {}
        row_like_keys = []
        for key, item in value.items():
            item_count = row_count(item)
            if item_count is None:
                subset[key] = item
                continue
            if item_count != int(expected_parent_rows):
                raise AssertionError(
                    f"{label}[{key!r}] has {item_count} rows/items, expected parent rows={expected_parent_rows}"
                )
            row_like_keys.append(key)
            subset[key] = subset_row_like(item, indices, expected_parent_rows, label=f"{label}[{key!r}]")
        if not row_like_keys:
            raise AssertionError(f"{label} is a dict but has no row-like values of parent length")
        return subset

    raise TypeError(f"{label} is not a supported row-like object: {type(value).__name__}")


def describe_pt_object(value) -> str:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return f"Tensor{tuple(value.shape)} dtype={value.dtype}"
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape} dtype={value.dtype}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        return "dict(" + ", ".join(f"{key}={describe_pt_object(item)}" for key, item in value.items()) + ")"
    return type(value).__name__


def default_csv_path(dataset_dir: Path) -> Path:
    return dataset_dir / f"{dataset_dir.name}.csv"


def default_parent_tokenized_cache_dir(parent_dir: Path) -> Path:
    candidates = [
        parent_dir / f"{parent_dir.name}_cache",
        parent_dir.parent / f"{parent_dir.name}_cache",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def output_tokenized_cache_dir(subset_dir: Path) -> Path:
    return subset_dir.parent / f"{subset_dir.name}_cache"


def resolve_subset_csv(subset_dir: Path, subset_csv: str | None) -> Path:
    if subset_csv is None:
        return default_csv_path(subset_dir)
    path = Path(subset_csv).expanduser()
    if path.is_absolute():
        return path
    return subset_dir / path


def load_original_indices(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    indices = np.asarray(value, dtype=int).reshape(-1)
    return indices


def numbered_value(path: Path) -> int | None:
    match = re.search(r"(?:^|_)(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def discover_nmuts(embedding_dir: Path) -> list[int]:
    values = []
    for path in embedding_dir.glob("*.pt"):
        if "embedding" not in path.stem or "nmut" not in path.stem:
            continue
        value = numbered_value(path)
        if value is not None:
            values.append(value)
    values = sorted(set(values))
    if not values:
        raise FileNotFoundError(f"no embedding chunks found in {embedding_dir}")
    return values


def filter_nmuts(values: list[int], min_nmut: int | None, max_nmut: int | None) -> list[int]:
    selected = []
    for value in values:
        if min_nmut is not None and int(value) < int(min_nmut):
            continue
        if max_nmut is not None and int(value) > int(max_nmut):
            continue
        selected.append(int(value))
    return selected


def expected_subset_indices(subset_df: pd.DataFrame, num_muts_col: str, min_nmut: int | None, max_nmut: int | None) -> np.ndarray:
    if num_muts_col not in subset_df.columns:
        if min_nmut is not None or max_nmut is not None:
            raise KeyError(f"cannot filter by mutation count; column {num_muts_col!r} is missing")
        return np.arange(len(subset_df), dtype=int)
    nmuts = subset_df[num_muts_col].astype(int).to_numpy()
    mask = np.ones(len(subset_df), dtype=bool)
    if min_nmut is not None:
        mask &= nmuts >= int(min_nmut)
    if max_nmut is not None:
        mask &= nmuts <= int(max_nmut)
    return np.where(mask)[0].astype(int)


def column_span(df: pd.DataFrame, first_col: str | None, last_col: str | None, *, label: str) -> list[str] | None:
    if first_col is None and last_col is None:
        return None
    if first_col is None or last_col is None:
        raise ValueError(f"{label}: both first and last position columns must be provided")
    columns = list(df.columns)
    if first_col not in columns or last_col not in columns:
        raise ValueError(f"{label}: could not find position columns {first_col!r} and {last_col!r}")
    si = columns.index(first_col)
    ei = columns.index(last_col)
    if ei < si:
        raise ValueError(f"{label}: last position column must appear after first position column")
    selected = columns[si : ei + 1]
    print(
        "[subset] position span: "
        f"{label} first={first_col} last={last_col} start_index={si} end_index={ei} count={len(selected)}"
    )
    print(f"[subset] position columns {label}: {selected}")
    return selected


def parse_subset_position_ranges(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    ranges: dict[str, tuple[str, str]] = {}
    if args.first_col_his2 is not None or args.last_col_his2 is not None:
        if args.first_col_his2 is None or args.last_col_his2 is None:
            raise ValueError("both --first_col_his2 and --last_col_his2 are required for his2 position slicing")
        ranges["his2"] = (args.first_col_his2, args.last_col_his2)
    if args.first_col_his5 is not None or args.last_col_his5 is not None:
        if args.first_col_his5 is None or args.last_col_his5 is None:
            raise ValueError("both --first_col_his5 and --last_col_his5 are required for his5 position slicing")
        ranges["his5"] = (args.first_col_his5, args.last_col_his5)
    for spec in args.subset_position_range:
        parts = str(spec).split(":")
        if len(parts) != 3:
            raise ValueError("--subset_position_range must be formatted as subset_name:first_col:last_col")
        ranges[parts[0]] = (parts[1], parts[2])
    return ranges


def subset_position_indices(
    parent_position_columns: list[str] | None,
    subset_position_columns: list[str] | None,
    *,
    subset_name: str,
) -> np.ndarray | None:
    if subset_position_columns is None:
        return None
    if parent_position_columns is None:
        raise ValueError(f"{subset_name}: subset position range was provided but parent position range is missing")
    parent_lookup = {column: index for index, column in enumerate(parent_position_columns)}
    missing = [column for column in subset_position_columns if column not in parent_lookup]
    if missing:
        raise AssertionError(
            f"{subset_name}: subset position columns are not present in parent position span: {missing[:10]}"
        )
    indices = np.asarray([parent_lookup[column] for column in subset_position_columns], dtype=int)
    if len(np.unique(indices)) != len(indices):
        raise AssertionError(f"{subset_name}: duplicate position indices detected")
    print(f"[subset] position columns within parent for {subset_name}: {subset_position_columns}")
    print(f"[subset] position indices within parent for {subset_name}: {indices.tolist()}")
    return indices


def find_numbered_file(directory: Path, prefixes: tuple[str, ...], k: int, label: str) -> Path:
    for prefix in prefixes:
        for name in (f"{prefix}_{k}.pt", f"{prefix}{k}.pt"):
            path = directory / name
            if path.is_file():
                return path
    candidates = [
        path
        for path in directory.glob("*.pt")
        if label in path.stem and "nmut" in path.stem and numbered_value(path) == int(k)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise FileNotFoundError(f"ambiguous {label} chunk for nmut={k} in {directory}: {candidates}")
    raise FileNotFoundError(f"missing {label} chunk for nmut={k} in {directory}")


def validate_subset_rows(
    parent_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    original_indices: np.ndarray,
    *,
    sequence_col: str,
    activity_col: str,
    num_muts_col: str,
    validate_samples: int,
    rng: np.random.Generator,
) -> None:
    print(
        "[subset] validating rows: "
        f"parent_rows={len(parent_df)} subset_rows={len(subset_df)} "
        f"original_indices={len(original_indices)}"
    )
    if len(subset_df) != len(original_indices):
        raise AssertionError(
            f"subset CSV has {len(subset_df)} rows but original indices has {len(original_indices)} rows"
        )
    if len(np.unique(original_indices)) != len(original_indices):
        raise AssertionError("original indices contain duplicates; output subset-local indices would be ambiguous")
    if np.any(original_indices < 0) or np.any(original_indices >= len(parent_df)):
        raise AssertionError("original indices point outside the parent dataframe")

    check_columns = [sequence_col]
    for column in (activity_col, num_muts_col):
        if column in parent_df.columns and column in subset_df.columns:
            check_columns.append(column)
    missing = [column for column in check_columns if column not in parent_df.columns or column not in subset_df.columns]
    if missing:
        raise KeyError(f"missing validation column(s): {missing}")

    n_check = min(int(validate_samples), len(subset_df))
    if n_check <= 0:
        return
    sampled_subset_rows = rng.choice(len(subset_df), size=n_check, replace=False)
    for subset_row in sampled_subset_rows:
        parent_row = int(original_indices[int(subset_row)])
        for column in check_columns:
            parent_value = parent_df.iloc[parent_row][column]
            subset_value = subset_df.iloc[int(subset_row)][column]
            if pd.isna(parent_value) and pd.isna(subset_value):
                continue
            if column in {activity_col, num_muts_col}:
                if np.isclose(float(parent_value), float(subset_value), equal_nan=True):
                    continue
                raise AssertionError(
                    f"subset row {subset_row} column {column!r} does not match parent row {parent_row}: "
                    f"{subset_value!r} != {parent_value!r}"
                )
            if str(parent_value) != str(subset_value):
                raise AssertionError(
                    f"subset row {subset_row} column {column!r} does not match parent row {parent_row}: "
                    f"{subset_value!r} != {parent_value!r}"
                )


def validate_parent_chunk(
    parent_df: pd.DataFrame,
    idx: np.ndarray,
    y_values: np.ndarray,
    k: int,
    *,
    activity_col: str,
    num_muts_col: str,
) -> None:
    if np.any(idx < 0) or np.any(idx >= len(parent_df)):
        raise AssertionError(f"indices_of_nmut_{k}.pt points outside the parent dataframe")
    if num_muts_col in parent_df.columns:
        nmuts = parent_df.iloc[idx][num_muts_col].astype(int).to_numpy()
        if not np.all(nmuts == int(k)):
            bad = idx[nmuts != int(k)][:10]
            raise AssertionError(f"indices_of_nmut_{k}.pt points to rows with other nmuts: {bad.tolist()}")
    if activity_col in parent_df.columns:
        parent_y = parent_df.iloc[idx][activity_col].astype(float).to_numpy()
        y_flat = np.asarray(y_values, dtype=float).reshape(-1)
        if len(parent_y) == len(y_flat) and not np.allclose(parent_y, y_flat, equal_nan=True, rtol=1e-4, atol=1e-6):
            raise AssertionError(f"y_values_of_nmut_{k}.pt does not match parent column {activity_col!r}")


def save_tensor(path: Path, tensor, *, skip_existing: bool, dry_run: bool) -> None:
    if path.exists() and skip_existing:
        print(f"[subset] skip existing {path}")
        return
    print(f"[subset] write {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(tensor, path)


def write_aggregated_chunks(
    output_model_dir: Path,
    chunk_paths: list[tuple[Path, Path, Path]],
    expected_indices: np.ndarray,
    *,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    output_emb_path = output_model_dir / "embeddings_all.pt"
    output_idx_path = output_model_dir / "indices_all.pt"
    output_y_path = output_model_dir / "y_values_all.pt"
    if dry_run:
        print(
            "[subset] aggregate dry-run: "
            f"model_dir={output_model_dir} expected_rows={len(expected_indices)}"
        )
        return
    if (
        skip_existing
        and output_emb_path.exists()
        and output_idx_path.exists()
        and output_y_path.exists()
    ):
        print(f"[subset] skip existing aggregate files in {output_model_dir}")
        return
    if not chunk_paths:
        if len(expected_indices) == 0:
            print(f"[subset] aggregate skipped: no expected rows for {output_model_dir}")
            return
        raise AssertionError(f"cannot aggregate {output_model_dir}; no chunk files were written")

    import torch

    emb_chunks = []
    idx_chunks = []
    y_chunks = []
    for emb_path, idx_path, y_path in chunk_paths:
        emb = to_cpu_tensor(torch_load(emb_path))
        idx = to_cpu_tensor(torch_load(idx_path)).long().reshape(-1)
        y = to_cpu_tensor(torch_load(y_path))
        if emb.shape[0] != idx.shape[0] or y.shape[0] != idx.shape[0]:
            raise AssertionError(
                f"aggregate chunk size mismatch in {output_model_dir}: "
                f"{emb_path.name} embeddings={tuple(emb.shape)} indices={tuple(idx.shape)} y={tuple(y.shape)}"
            )
        emb_chunks.append(emb)
        idx_chunks.append(idx)
        y_chunks.append(y)

    embeddings = torch.cat(emb_chunks, dim=0)
    indices = torch.cat(idx_chunks, dim=0)
    y_values = torch.cat(y_chunks, dim=0)
    order = torch.argsort(indices)
    embeddings = embeddings[order]
    indices = indices[order]
    y_values = y_values[order]

    expected_set = set(int(value) for value in expected_indices)
    observed = to_numpy(indices).astype(int).reshape(-1)
    observed_set = set(int(value) for value in observed)
    if len(observed) != len(observed_set):
        duplicated = sorted(value for value in observed_set if np.sum(observed == value) > 1)
        raise AssertionError(f"aggregate duplicate subset-local indices: {duplicated[:10]}")
    if observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        raise AssertionError(
            f"aggregate indices do not match expected rows; missing={missing[:10]} "
            f"(n={len(missing)}) extra={extra[:10]} (n={len(extra)})"
        )

    print(
        "[subset] aggregate write: "
        f"model_dir={output_model_dir} "
        f"embeddings_shape={tuple(embeddings.shape)} "
        f"indices_shape={tuple(indices.shape)} y_shape={tuple(y_values.shape)}"
    )
    save_tensor(output_emb_path, embeddings, skip_existing=skip_existing, dry_run=False)
    save_tensor(output_idx_path, indices, skip_existing=skip_existing, dry_run=False)
    save_tensor(output_y_path, y_values, skip_existing=skip_existing, dry_run=False)


def write_filtered_subset_dataframe(
    subset_dir: Path,
    subset_df: pd.DataFrame,
    expected_indices: np.ndarray,
    *,
    dry_run: bool,
) -> Path:
    output_path = subset_dir / f"new_{subset_dir.name}.csv"
    filtered_df = subset_df.iloc[expected_indices].copy()
    print(
        "[subset] filtered dataframe: "
        f"output={output_path} rows={len(filtered_df)} original_rows={len(subset_df)}"
    )
    if not dry_run:
        filtered_df.to_csv(output_path, index=False)
    return output_path


def subset_tokenized_cache(
    parent_cache_dir: Path,
    subset_dir: Path,
    original_indices: np.ndarray,
    expected_indices: np.ndarray,
    *,
    misc_name: str,
    parent_rows: int,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    parent_misc_dir = parent_cache_dir / misc_name
    if not parent_misc_dir.is_dir():
        raise FileNotFoundError(f"tokenized cache misc directory does not exist: {parent_misc_dir}")
    output_misc_dir = output_tokenized_cache_dir(subset_dir) / misc_name
    tokenized_paths = sorted(parent_misc_dir.glob("*.pt"))
    if not tokenized_paths:
        raise FileNotFoundError(f"no .pt tokenized cache files found in {parent_misc_dir}")

    subset_parent_indices = original_indices[expected_indices]
    print(
        "[subset] tokenized cache start: "
        f"parent_misc={parent_misc_dir} output_misc={output_misc_dir} "
        f"files={len(tokenized_paths)} selected_rows={len(subset_parent_indices)} parent_rows={parent_rows}"
    )
    for tokenized_path in tokenized_paths:
        output_path = output_misc_dir / tokenized_path.name
        if output_path.exists() and skip_existing:
            print(f"[subset] tokenized skip existing {output_path}")
            continue
        value = torch_load(tokenized_path)
        print(f"[subset] tokenized parent: file={tokenized_path.name} object={describe_pt_object(value)}")
        subset_value = subset_row_like(
            value,
            subset_parent_indices.astype(int),
            int(parent_rows),
            label=str(tokenized_path),
        )
        subset_count = row_count(subset_value)
        if subset_count != len(subset_parent_indices):
            raise AssertionError(
                f"subsetted tokenized file {tokenized_path.name} has {subset_count} rows/items, "
                f"expected {len(subset_parent_indices)}"
            )
        print(f"[subset] tokenized selected: file={tokenized_path.name} object={describe_pt_object(subset_value)}")
        save_tensor(output_path, subset_value, skip_existing=skip_existing, dry_run=dry_run)


def subset_embedding_model(
    model_dir: Path,
    output_model_dir: Path,
    parent_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    original_indices: np.ndarray,
    *,
    activity_col: str,
    num_muts_col: str,
    min_nmut: int | None,
    max_nmut: int | None,
    parent_position_count: int | None,
    position_indices: np.ndarray | None,
    write_aggregated: bool,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    selected_global_to_subset = {int(global_idx): subset_idx for subset_idx, global_idx in enumerate(original_indices)}
    expected_indices = expected_subset_indices(subset_df, num_muts_col, min_nmut, max_nmut)
    total_written = 0
    written_subset_indices: set[int] = set()
    aggregate_chunk_paths: list[tuple[Path, Path, Path]] = []
    print(
        "[subset] model start: "
        f"model={model_dir.name} subset_rows={len(subset_df)} "
        f"expected_rows={len(expected_indices)} "
        f"min_nmut={min_nmut} max_nmut={max_nmut} "
        f"parent_position_count={parent_position_count} "
        f"subset_position_count={None if position_indices is None else len(position_indices)} "
        f"unique_original_indices={len(selected_global_to_subset)}"
    )
    if position_indices is not None:
        if skip_existing:
            raise ValueError(
                "position slicing was requested together with --skip_existing. "
                "Re-run without --skip_existing so old full-position embedding chunks are overwritten."
            )
        print(f"[subset] model={model_dir.name} position indices used: {position_indices.tolist()}")

    nmuts_to_process = filter_nmuts(discover_nmuts(model_dir), min_nmut, max_nmut)
    print(f"[subset] model={model_dir.name} processing nmut chunks: {nmuts_to_process}")
    for k in nmuts_to_process:
        emb_path = find_numbered_file(model_dir, EMBEDDING_PREFIXES, k, "embedding")
        idx_path = find_numbered_file(model_dir, INDEX_PREFIXES, k, "indices")
        y_path = find_numbered_file(model_dir, Y_PREFIXES, k, "y")

        emb = to_cpu_tensor(torch_load(emb_path))
        idx_tensor = to_cpu_tensor(torch_load(idx_path))
        y_tensor = to_cpu_tensor(torch_load(y_path))
        idx = to_numpy(idx_tensor).astype(int).reshape(-1)
        y_values = to_numpy(y_tensor).reshape(-1)

        if emb.shape[0] != len(idx) or len(idx) != len(y_values):
            raise AssertionError(f"embedding/index/y length mismatch in {model_dir.name} nmut={k}")
        validate_parent_chunk(parent_df, idx, y_values, k, activity_col=activity_col, num_muts_col=num_muts_col)
        if parent_position_count is not None:
            if emb.ndim < 3:
                raise AssertionError(
                    f"{model_dir.name} nmut={k}: position slicing requested, but embeddings are not shaped (S, P, d); "
                    f"shape={tuple(emb.shape)}"
                )
            if int(emb.shape[1]) != int(parent_position_count):
                raise AssertionError(
                    f"{model_dir.name} nmut={k}: embedding position dimension P={int(emb.shape[1])} "
                    f"does not match parent position column count={int(parent_position_count)}"
                )
            print(
                "[subset] position assertion passed: "
                f"model={model_dir.name} nmut={k} embedding_P={int(emb.shape[1])} "
                f"parent_position_columns={int(parent_position_count)}"
            )
        print(
            "[subset] parent chunk: "
            f"model={model_dir.name} nmut={k} "
            f"embeddings_shape={tuple(emb.shape)} indices_shape={idx.shape} "
            f"y_shape={y_values.shape}"
        )

        selected_rows = [
            parent_chunk_row
            for parent_chunk_row, global_idx in enumerate(idx)
            if int(global_idx) in selected_global_to_subset
        ]
        if not selected_rows:
            print(f"[subset] model={model_dir.name} nmut={k}: selected 0 rows; skipping")
            continue

        import torch

        selected_rows = np.asarray(selected_rows, dtype=int)
        selected_global = idx[selected_rows]
        selected_subset = np.asarray([selected_global_to_subset[int(global_idx)] for global_idx in selected_global], dtype=int)
        order = np.argsort(selected_subset)
        selected_rows = selected_rows[order]
        selected_subset = selected_subset[order]

        if num_muts_col in subset_df.columns:
            subset_nmuts = subset_df.iloc[selected_subset][num_muts_col].astype(int).to_numpy()
            if not np.all(subset_nmuts == int(k)):
                bad = selected_subset[subset_nmuts != int(k)][:10]
                raise AssertionError(f"selected subset rows for nmut={k} have other nmuts: {bad.tolist()}")
        if activity_col in subset_df.columns:
            subset_y = subset_df.iloc[selected_subset][activity_col].astype(float).to_numpy()
            selected_row_tensor = torch.as_tensor(selected_rows, dtype=torch.long)
            selected_y = to_numpy(y_tensor[selected_row_tensor]).astype(float).reshape(-1)
            if not np.allclose(subset_y, selected_y, equal_nan=True, rtol=1e-4, atol=1e-6):
                raise AssertionError(f"selected y values for {model_dir.name} nmut={k} do not match subset CSV")

        subset_idx_tensor = torch.as_tensor(selected_subset, dtype=torch.long)
        selected_row_tensor = torch.as_tensor(selected_rows, dtype=torch.long)
        selected_emb = emb[selected_row_tensor]
        if position_indices is not None:
            position_tensor = torch.as_tensor(position_indices, dtype=torch.long)
            selected_emb = selected_emb[:, position_tensor, ...]
            if int(selected_emb.shape[1]) != len(position_indices):
                raise AssertionError(
                    f"{model_dir.name} nmut={k}: output position dimension={int(selected_emb.shape[1])} "
                    f"but expected {len(position_indices)}"
                )
        selected_y_tensor = y_tensor[selected_row_tensor]
        if selected_emb.shape[0] != len(selected_subset):
            raise AssertionError(
                f"{model_dir.name} nmut={k}: selected embedding rows={selected_emb.shape[0]} "
                f"but selected indices={len(selected_subset)}"
            )
        if selected_y_tensor.shape[0] != len(selected_subset):
            raise AssertionError(
                f"{model_dir.name} nmut={k}: selected y rows={selected_y_tensor.shape[0]} "
                f"but selected indices={len(selected_subset)}"
            )
        if np.any(selected_subset < 0) or np.any(selected_subset >= len(subset_df)):
            raise AssertionError(f"{model_dir.name} nmut={k}: output subset-local indices are out of bounds")
        if len(np.unique(selected_subset)) != len(selected_subset):
            raise AssertionError(f"{model_dir.name} nmut={k}: duplicate subset-local output indices")
        overlap = sorted(set(int(value) for value in selected_subset) & written_subset_indices)
        if overlap:
            raise AssertionError(
                f"{model_dir.name} nmut={k}: subset-local indices already written by another chunk: {overlap[:10]}"
            )
        written_subset_indices.update(int(value) for value in selected_subset)
        print(
            "[subset] selected chunk: "
            f"model={model_dir.name} nmut={k} "
            f"selected_rows={len(selected_subset)} "
            f"output_embeddings_shape={tuple(selected_emb.shape)} "
            f"output_indices_shape={tuple(subset_idx_tensor.shape)} "
            f"output_y_shape={tuple(selected_y_tensor.shape)}"
        )
        save_tensor(output_model_dir / emb_path.name, selected_emb, skip_existing=skip_existing, dry_run=dry_run)
        save_tensor(output_model_dir / idx_path.name, subset_idx_tensor, skip_existing=skip_existing, dry_run=dry_run)
        save_tensor(output_model_dir / y_path.name, selected_y_tensor, skip_existing=skip_existing, dry_run=dry_run)
        aggregate_chunk_paths.append((output_model_dir / emb_path.name, output_model_dir / idx_path.name, output_model_dir / y_path.name))
        total_written += len(selected_subset)
        print(
            "[subset] chunk done: "
            f"model={model_dir.name} nmut={k} "
            f"running_written={total_written}/{len(expected_indices)}"
        )

    if total_written != len(expected_indices):
        raise AssertionError(
            f"{model_dir.name}: wrote {total_written} embedding rows, expected {len(expected_indices)} subset rows "
            f"for min_nmut={min_nmut} max_nmut={max_nmut}"
        )
    missing_subset_indices = sorted(set(int(value) for value in expected_indices) - written_subset_indices)
    if missing_subset_indices:
        raise AssertionError(
            f"{model_dir.name}: missing embeddings for subset-local rows: {missing_subset_indices[:10]} "
            f"(total missing={len(missing_subset_indices)})"
        )
    if write_aggregated:
        write_aggregated_chunks(
            output_model_dir,
            aggregate_chunk_paths,
            expected_indices,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
    print(
        "[subset] model complete: "
        f"model={model_dir.name} wrote_total={total_written} expected_subset_rows={len(expected_indices)}"
    )


def discover_embedding_models(parent_embeddings_dir: Path, embedding_names: list[str] | None) -> list[Path]:
    if embedding_names:
        model_dirs = [parent_embeddings_dir / name for name in embedding_names]
    else:
        model_dirs = [path for path in sorted(parent_embeddings_dir.iterdir()) if path.is_dir()]
    missing = [path for path in model_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing embedding model directories: {missing}")
    if not model_dirs:
        raise FileNotFoundError(f"no embedding model directories found under {parent_embeddings_dir}")
    return model_dirs


def main() -> int:
    args = parse_args()
    if args.tokenized_only and args.embeddings_only:
        raise ValueError("--tokenized_only and --embeddings_only cannot both be set")
    do_embeddings = not bool(args.tokenized_only)
    do_tokenized = bool(args.copy_tokenized or args.tokenized_only) and not bool(args.embeddings_only)
    parent_dir = Path(args.parent_dir).expanduser().resolve()
    parent_csv = Path(args.parent_csv).expanduser().resolve() if args.parent_csv else default_csv_path(parent_dir)
    parent_embeddings_dir = (
        Path(args.parent_embeddings_dir).expanduser().resolve()
        if args.parent_embeddings_dir
        else parent_dir / "embeddings"
    )
    parent_tokenized_cache_dir = (
        Path(args.parent_tokenized_cache_dir).expanduser().resolve()
        if args.parent_tokenized_cache_dir
        else default_parent_tokenized_cache_dir(parent_dir)
    )
    parent_df = pd.read_csv(parent_csv)
    model_dirs = discover_embedding_models(parent_embeddings_dir, args.embedding_names) if do_embeddings else []
    parent_position_columns = column_span(
        parent_df,
        args.first_col_his,
        args.last_col_his,
        label=parent_dir.name,
    )
    subset_position_ranges = parse_subset_position_ranges(args)
    rng = np.random.default_rng(int(args.seed))

    print(f"[subset] parent csv: {parent_csv} rows={len(parent_df)}")
    print(f"[subset] modes: embeddings={do_embeddings} tokenized={do_tokenized}")
    if do_embeddings:
        print(f"[subset] parent embeddings: {parent_embeddings_dir}")
        print(f"[subset] embedding models: {[path.name for path in model_dirs]}")
    if do_tokenized:
        print(f"[subset] parent tokenized cache: {parent_tokenized_cache_dir}")

    for subset_dir_text in args.subset_dirs:
        subset_dir = Path(subset_dir_text).expanduser().resolve()
        subset_csv = resolve_subset_csv(subset_dir, args.subset_csv)
        original_indices_path = subset_dir / args.original_indices_name
        subset_df = pd.read_csv(subset_csv)
        original_indices = load_original_indices(original_indices_path)
        print(f"[subset] subset: {subset_dir} csv={subset_csv} rows={len(subset_df)}")
        if args.num_muts_col in subset_df.columns:
            counts = subset_df[args.num_muts_col].astype(int).value_counts().sort_index().to_dict()
            print(f"[subset] expected subset rows by {args.num_muts_col}: {counts}")
        expected_indices = expected_subset_indices(subset_df, args.num_muts_col, args.min_nmut, args.max_nmut)
        if args.num_muts_col in subset_df.columns:
            filtered_counts = subset_df.iloc[expected_indices][args.num_muts_col].astype(int).value_counts().sort_index().to_dict()
            print(
                "[subset] filtered subset rows by "
                f"{args.num_muts_col} for min_nmut={args.min_nmut} max_nmut={args.max_nmut}: {filtered_counts}"
            )
        subset_range = subset_position_ranges.get(subset_dir.name)
        subset_position_columns = None
        if subset_range is not None:
            subset_position_columns = column_span(
                subset_df,
                subset_range[0],
                subset_range[1],
                label=subset_dir.name,
            )
        position_indices = subset_position_indices(
            parent_position_columns,
            subset_position_columns,
            subset_name=subset_dir.name,
        )
        write_filtered_subset_dataframe(subset_dir, subset_df, expected_indices, dry_run=bool(args.dry_run))
        validate_subset_rows(
            parent_df,
            subset_df,
            original_indices,
            sequence_col=args.sequence_col,
            activity_col=args.activity_col,
            num_muts_col=args.num_muts_col,
            validate_samples=args.validate_samples,
            rng=rng,
        )
        print(f"[subset] validated {min(args.validate_samples, len(subset_df))} sampled subset rows")

        if do_tokenized:
            subset_tokenized_cache(
                parent_tokenized_cache_dir,
                subset_dir,
                original_indices,
                expected_indices,
                misc_name=args.tokenized_misc_name,
                parent_rows=len(parent_df),
                skip_existing=bool(args.skip_existing),
                dry_run=bool(args.dry_run),
            )
        if do_embeddings:
            output_embeddings_dir = subset_dir / args.output_embeddings_name
            for model_dir in model_dirs:
                subset_embedding_model(
                    model_dir,
                    output_embeddings_dir / model_dir.name,
                    parent_df,
                    subset_df,
                    original_indices,
                    activity_col=args.activity_col,
                    num_muts_col=args.num_muts_col,
                    min_nmut=args.min_nmut,
                    max_nmut=args.max_nmut,
                    parent_position_count=None if parent_position_columns is None else len(parent_position_columns),
                    position_indices=position_indices,
                    write_aggregated=bool(args.write_aggregated),
                    skip_existing=bool(args.skip_existing),
                    dry_run=bool(args.dry_run),
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
