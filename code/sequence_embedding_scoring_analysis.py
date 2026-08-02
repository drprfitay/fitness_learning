#!/usr/bin/env python3
"""
Standalone fitness-learning analysis for one-hot and precomputed PLM embeddings.

This is intentionally a copyable/debuggable script. It mirrors the 20 MLP
architectures and architecture-resolution behavior used in
src/experiments/run_scoring_experiment.py, but keeps the split layout explicit
for mutation-count and random-subset experiments.
"""

import argparse
import gc
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


DEFAULT_ARCHITECTURE_GRID = [
    {"name": "mlp_16", "hidden_layers": [16], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0, "batch_size": 64},
    {"name": "mlp_32", "hidden_layers": [32], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0, "batch_size": 64},
    {"name": "mlp_32_16", "hidden_layers": [32, 16], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0, "batch_size": 64},
    {"name": "mlp_16_16", "hidden_layers": [16, 16], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0, "batch_size": 64},
    {"name": "mlp_16_32", "hidden_layers": [16, 32], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0, "batch_size": 64},
    {"name": "mlp_128_32", "hidden_layers": [128, 32], "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_128_16", "hidden_layers": [128, 16], "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_32_128", "hidden_layers": [32, 128], "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_16_128", "hidden_layers": [16, 128], "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_128_32_16", "hidden_layers": [128, 32, 16], "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_64", "hidden_layers": [64], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0, "batch_size": 64},
    {"name": "mlp_128", "hidden_layers": [128], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_100_100", "hidden_layers": [100, 100], "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_256_128", "hidden_layers": [256, 128], "lr": 5e-4, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_256", "hidden_layers": [256], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_512", "hidden_layers": [512], "lr": 5e-4, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_1024", "hidden_layers": [1024], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.2, "batch_size": 64},
    {"name": "mlp_64_64", "hidden_layers": [64, 64], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_128_64", "hidden_layers": [128, 64], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_128_128", "hidden_layers": [128, 128], "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_256_256", "hidden_layers": [256, 256], "lr": 5e-4, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_512_256", "hidden_layers": [512, 256], "lr": 5e-4, "weight_decay": 1e-5, "dropout": 0.15, "batch_size": 64},
    {"name": "mlp_1024_256", "hidden_layers": [1024, 256], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.2, "batch_size": 64},
    {"name": "mlp_64_128", "hidden_layers": [64, 128], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_128_256", "hidden_layers": [128, 256], "lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_256_512", "hidden_layers": [256, 512], "lr": 5e-4, "weight_decay": 1e-5, "dropout": 0.15, "batch_size": 64},
    {"name": "mlp_64_128_64", "hidden_layers": [64, 128, 64], "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_128_256_128", "hidden_layers": [128, 256, 128], "lr": 5e-4, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 64},
    {"name": "mlp_256_512_256", "hidden_layers": [256, 512, 256], "lr": 5e-4, "weight_decay": 1e-5, "dropout": 0.15, "batch_size": 64},
    {"name": "mlp_512_1024_256", "hidden_layers": [512, 1024, 256], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.2, "batch_size": 64},
]


@dataclass
class LabelSpec:
    task_type: str
    cache_name: str
    y: np.ndarray


class MLPScorer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatureStore:
    def __init__(self, df, dataset_path, args, feature_kind, embedding_name=None):
        self.df = df
        self.dataset_path = Path(dataset_path)
        self.args = args
        self.feature_kind = feature_kind
        self.embedding_name = embedding_name
        self.model_name = "one_hot" if feature_kind == "onehot" else self._embedding_model_name()
        self.classifier_label = "one_hot" if feature_kind == "onehot" else f"{self.model_name}:{self.embedding_dir}"
        self._onehot = None
        self._all_embeddings = None
        self._embedding_cache = {}

    @property
    def embedding_dir(self):
        return self.dataset_path.parent / "embeddings" / str(self.embedding_name)

    def _embedding_model_name(self):
        prefix = "mean" if self.args.mean_embeddings else "flat"
        return f"{prefix}_{self.embedding_name}"

    def get(self, indices):
        indices = np.asarray(indices, dtype=int)
        if self.feature_kind == "onehot":
            return self._get_onehot(indices)
        if self.args.load_all:
            return self._get_all_embeddings(indices)
        return self._get_embeddings_chunked(indices)

    def iter_batches(self, indices, max_rows=None):
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return
        if max_rows is None or int(max_rows) < 0 or len(indices) <= int(max_rows):
            yield indices, self.get(indices)
            return
        max_rows = max(int(max_rows), 1)
        for start in range(0, len(indices), max_rows):
            batch_indices = indices[start : start + max_rows]
            yield batch_indices, self.get(batch_indices)

    def _get_onehot(self, indices):
        if self._onehot is None:
            relevant = get_relevant_mutation_columns(
                self.df, self.args.first_mutation_col, self.args.last_mutation_col
            )
            self._onehot = pd.get_dummies(self.df[relevant]).astype(np.float32).to_numpy()
            print(f"[features] built one-hot matrix shape={self._onehot.shape}")
        return self._onehot[indices]

    def _get_all_embeddings(self, indices):
        if self._all_embeddings is None:
            n_rows = len(self.df)
            arrays = []
            index_arrays = []
            for k in sorted(self.df[self.args.num_muts_colname].dropna().astype(int).unique()):
                emb, idx, _ = self._load_embedding_group(k)
                arrays.append(emb)
                index_arrays.append(idx)
            if not arrays:
                raise FileNotFoundError(f"no embedding files found in {self.embedding_dir}")
            dim = arrays[0].shape[1]
            out = np.full((n_rows, dim), np.nan, dtype=np.float32)
            for emb, idx in zip(arrays, index_arrays):
                if emb.shape[1] != dim:
                    raise ValueError("all loaded embedding chunks must have the same feature dimension")
                out[idx] = emb
            missing = np.where(~np.isfinite(out).all(axis=1))[0]
            if len(missing):
                print(f"[features] warning: {len(missing)} rows do not have loaded embeddings")
            self._all_embeddings = out
            print(f"[features] loaded all embeddings shape={self._all_embeddings.shape}")
        return self._all_embeddings[indices]

    def _get_embeddings_chunked(self, indices):
        if len(indices) == 0:
            return np.empty((0, 0), dtype=np.float32)
        nmuts = self.df.iloc[indices][self.args.num_muts_colname].astype(int).to_numpy()
        chunks = {}
        dim = None

        # LOGIC: Embeddings are heavy, and might not be saved together.
        # in this case, we might have embeddings grouped by the number of mutations
        # So when we are collected embeddings, we need to pluck them out 
        for k in sorted(set(int(value) for value in nmuts)):
            emb, idx, _ = self._load_embedding_group(k, cache=self.args.cache_embedding_chunks)
            local = {int(global_idx): row for row, global_idx in enumerate(idx)}
            wanted = [local[int(global_idx)] for global_idx in indices if int(self.df.iloc[global_idx][self.args.num_muts_colname]) == k]
            selected = emb[np.asarray(wanted, dtype=int)]
            chunks[k] = selected
            dim = selected.shape[1] if dim is None else dim
            if selected.shape[1] != dim:
                raise ValueError("embedding chunks returned inconsistent feature dimensions")
        out = np.empty((len(indices), dim), dtype=np.float32)
        offsets = {k: 0 for k in chunks}
        for out_row, global_idx in enumerate(indices):
            k = int(self.df.iloc[int(global_idx)][self.args.num_muts_colname])
            out[out_row] = chunks[k][offsets[k]]
            offsets[k] += 1
        return out

    def _load_embedding_group(self, k, cache=True):
        k = int(k)
        if cache and k in self._embedding_cache:
            return self._embedding_cache[k]

        # Here we want to validate that chunking for embedding was correct when first generated.
        # indices here should align exactly to indices in the sequence df, so we can always validate that
        # if this is group of K mutations, sequence_df[indices_of_nmut][nmuts]  == K across all entries
        # an additionaly assert is that sequence_df[indices_of_nmuts][activity] == y_values_of_nmut across all etnries 
        # (up to a rounding error that can happen from floating point inaccuracies movements in np/torch/pandas)
        emb_path = find_numbered_file(self.embedding_dir, ["embeddings_of_nmut", "embeddings_of_nmuts"], k)
        idx_path = find_numbered_file(self.embedding_dir, ["indices_of_nmut", "indices_of_nmuts"], k)
        y_path = find_numbered_file(self.embedding_dir, ["y_values_of_nmut", "y_values_of_nmuts"], k)
        emb = torch.load(emb_path, map_location="cpu")
        idx = torch.load(idx_path, map_location="cpu")
        y_values = torch.load(y_path, map_location="cpu")
        emb = tensor_to_numpy(emb)
        idx = tensor_to_numpy(idx).astype(int).reshape(-1)
        y_values = tensor_to_numpy(y_values).reshape(-1)

        # First make sure all are in same shape
        if emb.shape[0] != len(idx) or len(idx) != len(y_values):
            raise AssertionError(f"embedding/y/index length mismatch for nmut={k}")
        
        # Make sure all have the correct nmuts
        df_nmuts = self.df.iloc[idx][self.args.num_muts_colname].astype(int).to_numpy()
        if not np.all(df_nmuts == k):
            bad = idx[df_nmuts != k][:10]
            raise AssertionError(f"indices_of_nmut_{k}.pt points to rows with other nmuts: {bad.tolist()}")

        # Make sure all have correct activites
        df_y = self.df.iloc[idx][self.args.activity_column_name].astype(float).to_numpy()
        if not np.allclose(df_y, y_values.astype(float), equal_nan=True, rtol=1e-4, atol=1e-6):
            raise AssertionError(f"y_values_of_nmut_{k}.pt does not match {self.args.activity_column_name}")
        
        # Average embeddings if necessary
        emb = prepare_embedding_array(emb, mean_embeddings=self.args.mean_embeddings)
        result = (emb.astype(np.float32), idx, y_values)

        if cache:
            self._embedding_cache[k] = result
        if self.args.verbose_debug_prints:
            print(f"[features] loaded nmut={k}: embeddings={emb.shape} indices={idx.shape}")
        return result


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def prepare_embedding_array(emb, mean_embeddings=False):
    emb = np.asarray(emb)
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
    elif emb.ndim > 2:
        emb = emb.mean(axis=1) if mean_embeddings else emb.reshape(emb.shape[0], -1)
    return emb.astype(np.float32)


def find_numbered_file(directory, prefixes, k):
    directory = Path(directory)
    for prefix in prefixes:
        for name in (f"{prefix}_{k}.pt", f"{prefix}{k}.pt"):
            path = directory / name
            if path.is_file():
                return path
    tried = ", ".join(f"{prefix}_{k}.pt" for prefix in prefixes)
    raise FileNotFoundError(f"missing embedding chunk in {directory}; tried {tried}")


def get_relevant_mutation_columns(df, first_col, last_col):
    if first_col is None or last_col is None:
        raise ValueError("--first_mutation_col and --last_mutation_col are required for --onehot")
    columns = np.asarray(df.columns)
    start = np.where(columns == first_col)[0]
    end = np.where(columns == last_col)[0]
    if len(start) == 0 or len(end) == 0:
        raise ValueError(f"could not find mutation columns {first_col!r} and {last_col!r}")
    si = int(start[0])
    ei = int(end[0]) + 1
    if ei <= si:
        raise ValueError("--last_mutation_col must appear after --first_mutation_col")
    return df.columns[si:ei]


def make_label_spec(df, args):
    activity = df[args.activity_column_name]
    if args.regressor:
        return LabelSpec(
            task_type="regression",
            cache_name="regressor",
            y=activity.astype(float).to_numpy(),
        )

    if not args.classifier and args.classifier_percentile is None and args.classifier_value is None:
        raise ValueError("choose --regressor, --classifier, --classifier_percentile, or --classifier_value")

    if args.classifier_percentile is not None and args.classifier_value is not None:
        raise ValueError("use only one of --classifier_percentile or --classifier_value")
    if args.classifier_percentile is not None:
        threshold = float(np.nanpercentile(activity.astype(float).to_numpy(), args.classifier_percentile))
        labels = (activity.astype(float).to_numpy() >= threshold).astype(int)
        cache_name = f"classifier_percentile_{args.classifier_percentile:.3f}"
    elif args.classifier_value is not None:
        threshold = float(args.classifier_value)
        labels = (activity.astype(float).to_numpy() >= threshold).astype(int)
        cache_name = f"classifier_value_{threshold:.3f}"
    else:
        labels = pd.Categorical(activity).codes.astype(int)
        if np.any(labels < 0):
            raise ValueError("classifier labels contain NaN values")
        cache_name = "classifier"

    return LabelSpec(task_type="classification", cache_name=cache_name, y=labels)


def default_cache_path(dataset_path):
    dataset_path = Path(dataset_path)
    if dataset_path.suffix:
        return dataset_path.parent / "scoring_cache"
    return dataset_path / "scoring_cache"


def hard_refresh_label_cache(label_root, cache_path):
    label_root = Path(label_root).resolve()
    cache_path = Path(cache_path).resolve()
    if label_root == cache_path or cache_path not in label_root.parents:
        raise ValueError(f"refusing hard refresh outside cache root: {label_root}")
    if label_root.exists():
        print(f"[main] hard_refresh=True; deleting cache root: {label_root}")
        shutil.rmtree(label_root)
    else:
        print(f"[main] hard_refresh=True; cache root does not exist yet: {label_root}")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def count_available_splits(label_root):
    root = Path(label_root) / "resolved_splits"
    validation_root = root / "validation_splits"
    training_root = root / "training_splits"
    counts = {
        "full_validation_files": len(list((validation_root / "full").glob("*.json"))),
        "mutation_training_files": len(list((training_root / "by_mutation").glob("*.json"))),
        "mutation_less_than_K_files": len(list((validation_root / "by_mutation" / "less_than_K").glob("K_*/*.json"))),
        "mutation_equal_to_K_files": len(list((validation_root / "by_mutation" / "equal_to_K").glob("K_*/*.json"))),
        "mutation_specific_K_files": len(list((validation_root / "by_mutation" / "specific_K").glob("K_*/*.json"))),
        "random_training_files": len(list((training_root / "random").glob("*.json"))),
        "random_validation_files": len(list((validation_root / "random").glob("*.json"))),
        "random_internal_resamples": 0,
    }
    for path in (validation_root / "random").glob("*.json"):
        try:
            counts["random_internal_resamples"] += len(read_json(path).get("resamples") or [])
        except Exception as exc:
            print(f"[splits] warning: could not count resamples in {path}: {exc}")
    counts["total_json_files"] = sum(
        value for key, value in counts.items()
        if key.endswith("_files")
    )
    return counts


def print_split_counts(label_root, written_counts):
    available = count_available_splits(label_root)
    print(f"[splits] written this run: {written_counts}")
    print(f"[splits] available on disk: {available}")


def split_writes_requested(args):
    return bool(
        args.make_splits_only
        or args.refresh
        or args.hard_refresh
        or args.create_missing_splits
    )


def require_existing_splits(label_root, args):
    missing = []
    label_root = Path(label_root)
    validation_root = label_root / "resolved_splits" / "validation_splits"
    training_root = label_root / "resolved_splits" / "training_splits"

    if args.architecture_resolution == "full":
        if not any((validation_root / "full").glob("*.json")):
            missing.append(str(validation_root / "full" / "*.json"))

    if args.train_type == "random":
        for size in args.train_sizes:
            for iteration in range(args.niters):
                split_name = f"split_{int(size)}_{iteration + 1:03d}.json"
                train_path = training_root / "random" / split_name
                if not train_path.exists():
                    missing.append(str(train_path))
                if args.architecture_resolution == "random_internal":
                    val_path = validation_root / "random" / split_name
                    if not val_path.exists():
                        missing.append(str(val_path))

    if args.architecture_resolution in {"less_than_K", "equal_to_K"}:
        if args.min_muts is None or args.max_muts is None:
            missing.append(f"{args.architecture_resolution} requires --min_muts/--max_muts split metadata")
        else:
            for k in range(int(args.min_muts), int(args.max_muts)):
                if args.architecture_resolution == "equal_to_K":
                    path = validation_root / "by_mutation" / "equal_to_K" / f"K_{k}" / "train_le_K_validate_K_plus_1.json"
                    if not path.exists():
                        missing.append(str(path))
                else:
                    path = validation_root / "by_mutation" / "less_than_K" / f"K_{k}"
                    if not any(path.glob("*.json")):
                        missing.append(str(path / "*.json"))

    if args.architecture_resolution == "specific_K":
        path = validation_root / "by_mutation" / "specific_K" / f"K_{int(args.specific_k)}"
        if not any(path.glob("*.json")):
            missing.append(str(path / "*.json"))

    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(
            "required split files are missing and split writing was not requested.\n"
            f"{preview}{more}\n"
            "Create them explicitly with --create_missing_splits, --make_splits_only, "
            "--refresh, or --hard-refresh."
        )

    print("[splits] existing required split files found; not creating or updating splits")
    print(f"[splits] available on disk: {count_available_splits(label_root)}")


def random_split(indices, train_fraction, rng):
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2:
        return indices.tolist(), []
    order = rng.permutation(indices)
    n_train = int(round(len(indices) * float(train_fraction)))
    n_train = min(max(n_train, 1), len(indices) - 1)
    return order[:n_train].astype(int).tolist(), order[n_train:].astype(int).tolist()


def random_internal_fraction_plan(pool_size, args):
    fractions = [float(value) for value in args.random_internal_validation_fraction_split]
    if not fractions:
        raise ValueError("--random_internal_validation_fraction_split cannot be empty")

    min_val_points = int(args.random_internal_min_val_points)
    use_only_half = False
    for fraction in fractions:
        if not math.isclose(fraction, 0.75, rel_tol=0.0, abs_tol=1e-9):
            continue
        n_train = int(round(int(pool_size) * fraction))
        n_train = min(max(n_train, 1), max(int(pool_size) - 1, 1))
        n_val = int(pool_size) - n_train
        if n_val < min_val_points:
            use_only_half = True
            break

    if use_only_half:
        return [(0.5, int(args.validation_niters) * len(fractions))]
    return [(fraction, int(args.validation_niters)) for fraction in fractions]


def create_or_update_splits(df, label_root, args):
    print(f"[splits] resolving splits under {label_root}")
    print(
        "[splits] flags: "
        f"refresh={bool(args.refresh)} hard_refresh={bool(args.hard_refresh)} "
        f"create_missing_splits={bool(args.create_missing_splits)} "
        f"make_splits_only={bool(args.make_splits_only)}"
    )
    print(
        "[splits] random_internal defaults: "
        f"fractions={list(args.random_internal_validation_fraction_split)} "
        f"niters_per_fraction={int(args.validation_niters)} "
        f"min_val_points={int(args.random_internal_min_val_points)}"
    )
    rng = np.random.default_rng(args.random_seed)
    all_indices = np.arange(len(df), dtype=int)
    stats = {
        "full_validation_files": 0,
        "mutation_training_files": 0,
        "mutation_less_than_K_files": 0,
        "mutation_equal_to_K_files": 0,
        "mutation_specific_K_files": 0,
        "random_training_files": 0,
        "random_validation_files": 0,
        "random_internal_resamples": 0,
    }


    # LOGIC: Full validation splits are created to bootstrap the best architecture without the prior
    # knowledge of what exactly would be the train or test. This is a complementry to train-validation 
    # strategies in which you do explicitly account for the prior knowledge of what is the train-test.
    # So for example, here we are simply testing what would be the best architecture across the entire dataset,
    # whereas in other places, we will be interested in the best architecture given the partial sample we have made.
    # IE: best architecture to train on 1-muts, given sampled only 1-muts, and tested on >= 2 muts.
    # best architecture to train on 1-muts, validated on 2-muts, tested on >= 3 muts
    
    full_dir = label_root / "resolved_splits" / "validation_splits" / "full"
    if args.refresh or not any(full_dir.glob("*.json")):
        for fraction in args.validation_fraction_split_full:
            for iteration in range(args.validation_niters_full):
                train_idx, val_idx = random_split(all_indices, fraction, rng)

                # Assert to ensure no index leakage between train and validation sets
                assert np.isin(train_idx, val_idx).sum() == 0, "Train and val index leakage detected"
                assert np.isin(val_idx, train_idx).sum() == 0, "val and train index leakage detected"

                write_json(
                    full_dir / f"fraction_{fraction:.3f}_iter_{iteration + 1:03d}.json",
                    {
                        "mode": "full",
                        "train_fraction": float(fraction),
                        "iteration": int(iteration + 1),
                        "train_indices": train_idx,
                        "val_indices": val_idx,
                    },
                )
                stats["full_validation_files"] += 1

    if args.min_muts is not None and args.max_muts is not None:
        mutation_stats = write_mutation_splits(df, label_root, args, rng)
        for key, value in mutation_stats.items():
            stats[key] += value

    if args.train_sizes:
        random_stats = write_random_splits(df, label_root, args, rng)
        for key, value in random_stats.items():
            stats[key] += value

    print_split_counts(label_root, stats)


def write_mutation_splits(df, label_root, args, rng):
    nmuts = df[args.num_muts_colname].astype(int).to_numpy()
    validation_root = label_root / "resolved_splits" / "validation_splits" / "by_mutation"
    training_root = label_root / "resolved_splits" / "training_splits" / "by_mutation"
    stats = {
        "mutation_training_files": 0,
        "mutation_less_than_K_files": 0,
        "mutation_equal_to_K_files": 0,
        "mutation_specific_K_files": 0,
    }

    # LOGIC: Here we want to create different validation sets based on the train sets.
    # One time we would be intrested when trainining on <= k and testing on k to:
    # 1. train on <= k ; validate on <= k; test on > k
    # 2. train on < k ; validate on == k; test on > k.
    # 3. train on == k ; validate on == k; test on < k or > k
    # Train-test would be identical in both these case. 
    # You will always test on > k. however: 
    # in the first case you will train on <= k, and validate on <= k
    # in the second case you will train on < k and validate on = k
    for k in range(int(args.min_muts), int(args.max_muts) + 1):
        train_le_k = np.where(nmuts <= k)[0].astype(int)
        test_gt_k = np.where(nmuts > k)[0].astype(int)

        # Assert to ensure no index leakage between train and test sets
        assert np.isin(train_le_k, test_gt_k).sum() == 0, "Train <= k and test > k index leakage detected"
        assert np.isin(test_gt_k, train_le_k).sum() == 0, "test > k and train <= k index leakage detected"

        # Here we set the overall train-test indices.
        train_path = training_root / f"train_on_le_{k}.json"
        if args.refresh or not train_path.exists():
            write_json(
                train_path,
                {
                    "mode": "mutation",
                    "k": int(k),
                    "train_indices": train_le_k.tolist(),
                    "test_indices": test_gt_k.tolist(),
                },
            )
            stats["mutation_training_files"] += 1

        # here we set validation paradigm #1. If we train <= k and validate on <= k, 
        # these sets are rather stochastic, so we resample train <= k, validate <= k.
        # Eventually we will resolve the best architecture given this bootstrap,
        # and then train <= k and test on > K
        less_dir = validation_root / "less_than_K" / f"K_{k}"
        if args.refresh or not any(less_dir.glob("*.json")):
            for iteration in range(args.validation_niters):
                train_idx, val_idx = random_split(train_le_k, args.validation_fraction_split, rng)

                # Assert to ensure no index leakage between train and validation sets when <= k
                assert np.isin(train_idx, val_idx).sum() == 0, "Train <= k and validate <= k index leakage detected"
                assert np.isin(val_idx, train_idx).sum() == 0, "validate <= k and train <= k index leakage detected"

                write_json(
                    less_dir / f"iter_{iteration + 1:03d}.json",
                    {
                        "mode": "less_than_K",
                        "k": int(k),
                        "train_fraction": float(args.validation_fraction_split),
                        "iteration": int(iteration + 1),
                        "train_indices": train_idx,
                        "val_indices": val_idx,
                    },
                )
                stats["mutation_less_than_K_files"] += 1


        # here we set validation paradigm #2. If we train < k and validate on == k, 
        # these sets are deterministic.
        # The indices for train < k are fixed, and indices for validate == k are also fixed
        # Eventually, we will resolve the best architecture for train < k validate on k. and we will 
        # once train with that architecture on < k and test on > k
        # then train with that architecture on <= k and test on > k
        # keep in mind, in both these cases, we validated on == K just to figure out the best training strategy.
        eq_dir = validation_root / "equal_to_K" / f"K_{k}"
        eq_path = eq_dir / "train_le_K_validate_K_plus_1.json"
        if args.refresh or not eq_path.exists():
            val_eq_k = np.where(nmuts == k + 1)[0].astype(int).tolist()
            write_json(
                eq_path,
                {
                    "mode": "equal_to_K",
                    "k": int(k),
                    "interpretation": "train on <= K, validate on == K + 1",
                    "train_indices": train_le_k.tolist(),
                    "val_indices": val_eq_k,
                },
            )
            stats["mutation_equal_to_K_files"] += 1

            # Assert to ensure no index leakage between train and validation sets when <= k
            assert np.isin(train_le_k, val_eq_k).sum() == 0, "Train <= k and validate == k index leakage detected"
            assert np.isin(val_eq_k, train_le_k).sum() == 0, "validate == k and train <= k index leakage detected"

        # Here we set validation paradigm #3 if we train on == k and validate on ==k.
        # these sets are also stochastic
        exact_k = np.where(nmuts == k)[0].astype(int)
        specific_dir = validation_root / "specific_K" / f"K_{k}"
        if args.refresh or not any(specific_dir.glob("*.json")):
            for iteration in range(args.validation_niters):
                train_idx, val_idx = random_split(exact_k, args.validation_fraction_split, rng)

                # Assert to ensure no index leakage between train and validation sets when == k
                assert np.isin(train_idx, val_idx).sum() == 0, "Train == k and validate == k index leakage detected"
                assert np.isin(val_idx, train_idx).sum() == 0, "validate == k and train == k index leakage detected"

                write_json(
                    specific_dir / f"iter_{iteration + 1:03d}.json",
                    {
                        "mode": "specific_K",
                        "k": int(k),
                        "train_fraction": float(args.validation_fraction_split),
                        "iteration": int(iteration + 1),
                        "train_indices": train_idx,
                        "val_indices": val_idx,
                    },
                )
                stats["mutation_specific_K_files"] += 1

    return stats


def write_random_splits(df, label_root, args, rng):
    all_indices = np.arange(len(df), dtype=int)
    training_dir = label_root / "resolved_splits" / "training_splits" / "random"
    validation_dir = label_root / "resolved_splits" / "validation_splits" / "random"
    stats = {
        "random_training_files": 0,
        "random_validation_files": 0,
        "random_internal_resamples": 0,
    }

    for size in args.train_sizes:
        size = int(size)
        if size <= 0 or size >= len(df):
            raise ValueError(f"random train size must be in [1, n_rows - 1], got {size}")
        for iteration in range(args.niters):
            split_name = f"split_{size}_{iteration + 1:03d}.json"
            train_path = training_dir / split_name
            if args.refresh or not train_path.exists():
                train_idx = rng.choice(all_indices, size=size, replace=False).astype(int)
                test_idx = np.setdiff1d(all_indices, train_idx, assume_unique=False).astype(int)

                # Assert to ensure no index leakage between train and test sets
                assert np.isin(train_idx, test_idx).sum() == 0, "Train and test index leakage detected"
                assert np.isin(test_idx, train_idx).sum() == 0, "Test and train index leakage detected"
         
                write_json(
                    train_path,
                    {
                        "mode": "random",
                        "train_size": int(size),
                        "iteration": int(iteration + 1),
                        "train_indices": train_idx.tolist(),
                        "test_indices": test_idx.tolist(),
                    },
                )
                stats["random_training_files"] += 1
            train_payload = read_json(train_path)
            val_path = validation_dir / split_name

            # LOGIC: When we are training on a random subset, the chosen architecture / epochs will be either based on
            # the best resolved architecture from boostraps ("full resolution") or from internal train - validation of that split.
            # So for example, if for the specific iteraiton, we sampled 50 train points
            # (stored in train_payload["train_indices"]) we will also create a train-validation scheme based on
            # re-sampling of those same 50 data points. Keep in mind, this is why we use train_pool here.
            if args.refresh or not val_path.exists():
                resamples = []
                train_pool = np.asarray(train_payload["train_indices"], dtype=int)
                resample_index = 1

                # LOGIC: If we're training on 10 data points, 0.75 - 0.25 train - val makes no sense, but 0.5 - 0.5 does, 
                # so the idea here is to resolve how many splits per fraction of train-val to make. 
                # if |val| <= 5 points, we can have 0.5 X 10, if |val| >= 5, we will have 5 X 0.5, + 5 X 0.75 
                fraction_plan = random_internal_fraction_plan(len(train_pool), args) 
                for fraction, n_iterations in fraction_plan:
                    for val_iter in range(n_iterations):
                        inner_train, inner_val = random_split(train_pool, fraction, rng)


                        # Assert to ensure no index leakage between train and validation sets
                        assert np.isin(inner_train, inner_val).sum() == 0, "Train and val index leakage detected"
                        assert np.isin(inner_val, inner_train).sum() == 0, "val and train index leakage detected"

                        resamples.append(
                            {
                                "train_fraction": float(fraction),
                                "iteration": int(resample_index),
                                "fraction_iteration": int(val_iter + 1),
                                "train_indices": inner_train,
                                "val_indices": inner_val,
                            }
                        )
                        resample_index += 1
                write_json(
                    val_path,
                    {
                        "mode": "random_internal",
                        "source_training_split": str(train_path),
                        "train_size": int(size),
                        "iteration": int(iteration + 1),
                        "fraction_plan": [
                            {"train_fraction": float(fraction), "n_iterations": int(n_iterations)}
                            for fraction, n_iterations in fraction_plan
                        ],
                        "resamples": resamples,
                    },
                )
                stats["random_validation_files"] += 1
                stats["random_internal_resamples"] += len(resamples)

    return stats


# Used in the context of resolving best configuration
def load_resamples(label_root, mode, k=None, split_name=None):
    base = label_root / "resolved_splits" / "validation_splits"

    # validation indices based on all data
    if mode == "full":
        return [read_json(path) for path in sorted((base / "full").glob("*.json"))]

    # validation indices based on <= K; == K; specific K
    if mode in {"less_than_K", "equal_to_K", "specific_K"}:
        if k is None:
            raise ValueError(f"{mode} requires k")
        path = base / "by_mutation" / mode / f"K_{int(k)}"

        if mode == "equal_to_K":
            return [read_json(path / "train_le_K_validate_K_plus_1.json")]

        return [read_json(item) for item in sorted(path.glob("*.json"))]

    # Resampling random train-test validation indices
    if mode == "random_internal":
        if split_name is None:
            raise ValueError("random_internal requires split_name")
        payload = read_json(base / "random" / split_name)
        return payload["resamples"]

    raise ValueError(f"unknown architecture resolution mode: {mode}")


def normalize_train_test(X_train, X_other, enabled):
    if not enabled:
        return X_train, X_other
    mean, std = normalization_params(X_train)
    return apply_normalization(X_train, mean, std), apply_normalization(X_other, mean, std)


def normalization_params(X_train):
    mean = np.nanmean(X_train, axis=0, keepdims=True)
    std = np.nanstd(X_train, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_normalization(X, mean, std):
    return (X - mean) / std


def stream_eval_in_case_of_limited_memory(feature_store, indices, args):
    if feature_store.feature_kind != "embedding":
        return False
    if args.load_all:
        return False
    max_rows = int(args.maximum_embeddings_to_load)
    return max_rows >= 0 and len(indices) > max_rows


def eval_batch_size(args):
    max_rows = int(args.maximum_embeddings_to_load)
    if max_rows < 0:
        return None
    return max(max_rows, 1)


def set_random_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def as_tensor(value, name):
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except Exception as exc:
        raise TypeError(f"could not convert {name} to tensor") from exc


def resolve_device(device):
    if device is None or str(device).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def train_scorer_with_checkpoints(
    X_train,
    y_train,
    y_val,
    task_type,
    config,
    checkpoint_epochs,
    device,
    random_seed,
    X_val=None,
    feature_store=None,
    val_indices=None,
    normalize_eval=False,
    train_mean=None,
    train_std=None,
    eval_max_rows=None,
    precision_k=100,
    selection_metric=None,
):
    set_random_seed(random_seed)
    device = torch.device(resolve_device(device))
    x_train = as_tensor(X_train, "X_train").float().to(device)
    x_val = None if X_val is None else as_tensor(X_val, "X_val").float().to(device)
    y_train_tensor = as_tensor(y_train, "y_train")
    y_val_tensor = as_tensor(y_val, "y_val")

    if task_type == "classification":
        y_train_device = y_train_tensor.long().to(device)
        output_dim = int(torch.max(torch.cat([y_train_tensor.long(), y_val_tensor.long()])).item()) + 1
        loss_fn = nn.CrossEntropyLoss()
    else:
        y_train_device = y_train_tensor.float().reshape(-1, 1).to(device)
        output_dim = 1
        loss_fn = nn.MSELoss()

    model = MLPScorer(
        input_dim=x_train.shape[1],
        output_dim=output_dim,
        hidden_layers=config.get("hidden_layers", [64]),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    batch_size = int(config.get("batch_size", 64))
    checkpoint_epochs = [int(epoch) for epoch in checkpoint_epochs]
    checkpoint_set = set(checkpoint_epochs)
    checkpoint_scores = []
    checkpoint_metrics = []

    for epoch in range(1, max(checkpoint_epochs) + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(x_train[batch_idx])
            loss = loss_fn(output, y_train_device[batch_idx])
            loss.backward()
            optimizer.step()
        if epoch in checkpoint_set:
            model.eval()
            if x_val is not None:
                predictions = predict_array(model, x_val, task_type)
            else:
                # LOGIC: in case we cannot load all emeddings at once as they are heavy, predict
                # batch by batch, aggreagate and then evaluate all together.
                predictions = predict_feature_store_batches(
                    model,
                    feature_store,
                    val_indices,
                    task_type,
                    device,
                    max_rows=eval_max_rows,
                    normalize=normalize_eval,
                    mean=train_mean,
                    std=train_std,
                )
            
            # eval after aggregatign or predicting right away for all
            metrics = evaluate_predictions(
                y_val_tensor.detach().cpu().numpy(),
                predictions,
                task_type,
                precision_k=precision_k,
            )
            checkpoint_metrics.append(metrics)
            checkpoint_scores.append(primary_score(metrics, task_type, selection_metric=selection_metric))
    return {"checkpoint_scores": checkpoint_scores, "checkpoint_metrics": checkpoint_metrics}


def train_final_predictor(X_train, y_train, X_test, y_test, task_type, config, epochs, device, random_seed, precision_k=100):
    set_random_seed(random_seed)
    device = torch.device(resolve_device(device))
    x_train = as_tensor(X_train, "X_train").float().to(device)
    x_test = as_tensor(X_test, "X_test").float().to(device)
    y_train_tensor = as_tensor(y_train, "y_train")
    y_test_tensor = as_tensor(y_test, "y_test")

    if task_type == "classification":
        y_train_device = y_train_tensor.long().to(device)
        output_dim = int(torch.max(torch.cat([y_train_tensor.long(), y_test_tensor.long()])).item()) + 1
        loss_fn = nn.CrossEntropyLoss()
    else:
        y_train_device = y_train_tensor.float().reshape(-1, 1).to(device)
        output_dim = 1
        loss_fn = nn.MSELoss()

    model = MLPScorer(
        input_dim=x_train.shape[1],
        output_dim=output_dim,
        hidden_layers=config.get("hidden_layers", [64]),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    batch_size = int(config.get("batch_size", 64))

    for _ in range(int(epochs)):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(x_train[batch_idx])
            loss = loss_fn(output, y_train_device[batch_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(x_test)
        if task_type == "classification":
            predictions = torch.softmax(output, dim=1).detach().cpu().numpy()
        else:
            predictions = output.reshape(-1).detach().cpu().numpy()
    metrics = evaluate_predictions(
        y_test_tensor.detach().cpu().numpy(),
        predictions,
        task_type,
        precision_k=precision_k,
    )
    return {"metrics": metrics, "predictions": predictions}


def predict_array(model, x_tensor, task_type):
    model.eval()
    with torch.no_grad():
        output = model(x_tensor)
        if task_type == "classification":
            return torch.softmax(output, dim=1).detach().cpu().numpy()
        return output.reshape(-1).detach().cpu().numpy()


# LOGIC:
# under some cases embeddings will be extremly heavy and we will not be able to load all of them together all at once.
# in this case we will first PREDICT for all batches, aggregate all predcitions
# and only then evaluate.
def predict_feature_store_batches(
    model,
    feature_store,
    indices,
    task_type,
    device,
    *,
    max_rows=None,
    normalize=False,
    mean=None,
    std=None,
):
    if feature_store is None:
        raise ValueError("feature_store is required for batched prediction")
    indices = np.asarray(indices, dtype=int)
    predictions = []
    device = torch.device(resolve_device(device))
    for batch_indices, X_batch in feature_store.iter_batches(indices, max_rows=max_rows):
        if normalize:
            X_batch = apply_normalization(X_batch, mean, std)
        x_batch = as_tensor(X_batch, "X_batch").float().to(device)
        predictions.append(predict_array(model, x_batch, task_type))
        del X_batch, x_batch
        gc.collect()
    if not predictions:
        return np.asarray([])
    return np.concatenate(predictions, axis=0)


def train_final_predictor_from_feature_store(
    feature_store,
    train_idx,
    test_idx,
    y,
    task_type,
    config,
    epochs,
    device,
    random_seed,
    args,
):
    trained = train_predictor_from_feature_store(
        feature_store=feature_store,
        train_idx=train_idx,
        y=y,
        task_type=task_type,
        config=config,
        epochs=epochs,
        device=device,
        random_seed=random_seed,
        args=args,
    )
    return evaluate_trained_predictor_from_feature_store(
        trained,
        feature_store=feature_store,
        test_idx=test_idx,
        y=y,
        task_type=task_type,
        device=device,
        args=args,
    )


def train_predictor_from_feature_store(
    feature_store,
    train_idx,
    y,
    task_type,
    config,
    epochs,
    device,
    random_seed,
    args,
):
    train_idx = np.asarray(train_idx, dtype=int)
    X_train = feature_store.get(train_idx)
    normalize = args.normalize_embeddings and feature_store.feature_kind == "embedding"
    if normalize:
        mean, std = normalization_params(X_train)
        X_train = apply_normalization(X_train, mean, std)
    else:
        mean, std = None, None
    result = train_predictor_model(
        X_train=X_train,
        y_train=y[train_idx],
        task_type=task_type,
        config=config,
        epochs=epochs,
        device=device,
        random_seed=random_seed,
        output_dim=infer_output_dim(task_type, y),
    )
    result.update({"normalize": normalize, "mean": mean, "std": std})
    return result


def evaluate_trained_predictor_from_feature_store(
    trained,
    feature_store,
    test_idx,
    y,
    task_type,
    device,
    args,
):
    test_idx = np.asarray(test_idx, dtype=int)
    predictions = predict_feature_store_batches(
        trained["model"],
        feature_store,
        test_idx,
        task_type,
        device,
        max_rows=eval_batch_size(args),
        normalize=trained["normalize"],
        mean=trained["mean"],
        std=trained["std"],
    )
    metrics = evaluate_predictions(y[test_idx], predictions, task_type, precision_k=args.precision_k)
    return {"metrics": metrics, "predictions": predictions}


def infer_output_dim(task_type, y):
    if task_type == "classification":
        labels = np.asarray(y, dtype=int).reshape(-1)
        return int(np.max(labels)) + 1
    return 1


def train_predictor_model(
    X_train,
    y_train,
    task_type,
    config,
    epochs,
    device,
    random_seed,
    output_dim=None,
):
    set_random_seed(random_seed)
    device = torch.device(resolve_device(device))
    x_train = as_tensor(X_train, "X_train").float().to(device)
    y_train_tensor = as_tensor(y_train, "y_train")

    if task_type == "classification":
        y_train_device = y_train_tensor.long().to(device)
        output_dim = int(output_dim if output_dim is not None else torch.max(y_train_tensor.long()).item() + 1)
        loss_fn = nn.CrossEntropyLoss()
    else:
        y_train_device = y_train_tensor.float().reshape(-1, 1).to(device)
        output_dim = 1
        loss_fn = nn.MSELoss()

    model = MLPScorer(
        input_dim=x_train.shape[1],
        output_dim=output_dim,
        hidden_layers=config.get("hidden_layers", [64]),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    batch_size = int(config.get("batch_size", 64))

    for _ in range(int(epochs)):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(x_train[batch_idx])
            loss = loss_fn(output, y_train_device[batch_idx])
            loss.backward()
            optimizer.step()
    return {"model": model}


def evaluate_predictions(y_true, y_pred, task_type, precision_k=100):
    y_true = np.asarray(y_true).reshape(-1)
    metrics = {
        "pearson": np.nan,
        "spearman": np.nan,
        "spearman_p_value": np.nan,
        "mse": np.nan,
        "roc_auc": np.nan,
        "precision": np.nan,
        "precision_at_k": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "accuracy": np.nan,
    }
    if len(y_true) == 0:
        return metrics

    if task_type == "regression":
        values = np.asarray(y_pred).reshape(-1)
        metrics["mse"] = float(np.mean((y_true.astype(float) - values.astype(float)) ** 2))
        metrics["pearson"] = pearson(y_true, values)
        metrics["spearman"], metrics["spearman_p_value"] = spearman_with_pvalue(y_true, values)
        return metrics

    probabilities = np.asarray(y_pred)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    pred_labels = np.argmax(probabilities, axis=1)
    labels_present = np.unique(y_true)
    average = "binary" if len(labels_present) <= 2 and probabilities.shape[1] <= 2 else "macro"
    metrics["accuracy"] = float(accuracy_score(y_true, pred_labels))
    metrics["precision"] = float(precision_score(y_true, pred_labels, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, pred_labels, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, pred_labels, average=average, zero_division=0))
    metrics["roc_auc"] = roc_auc(y_true, probabilities)
    ranking_score = probabilities[:, 1] if probabilities.shape[1] >= 2 else pred_labels
    metrics["precision_at_k"] = precision_at_k(y_true, ranking_score, k=precision_k)
    metrics["spearman"], metrics["spearman_p_value"] = spearman_with_pvalue(y_true, ranking_score)
    return metrics


def pearson(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_with_pvalue(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) < 2:
        return float("nan"), float("nan")
    if spearmanr is None:
        return float(pd.Series(a).corr(pd.Series(b), method="spearman")), float("nan")
    result = spearmanr(a, b, nan_policy="omit")
    return float(result.correlation), float(result.pvalue)


def roc_auc(y_true, probabilities):
    try:
        probabilities = np.asarray(probabilities)
        labels_present = np.unique(y_true)
        if len(labels_present) < 2:
            return float("nan")
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(y_true, probabilities, multi_class="ovr"))
    except Exception:
        return float("nan")


def precision_at_k(y_true, scores, k=100):
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(y_true) == 0:
        return float("nan")
    k = min(int(k), len(y_true))
    if k <= 0:
        return float("nan")
    ranked = np.argsort(-scores)[:k]
    return float(np.mean(y_true[ranked] == 1))


def primary_score(metrics, task_type, selection_metric=None):
    metric = selection_metric
    if metric is None:
        metric = "spearman" if task_type == "regression" else "roc_auc"
    if metric == "negative_mse":
        return -float(metrics["mse"])
    return float(metrics[metric])


def architecture_selection_metric(task_type, resamples):
    if task_type != "regression":
        return "roc_auc"
    for resample in resamples:
        val_indices = resample.get("val_indices") or []

        # This is very rare, only in cases which we want to select based on validate == K and we don't have enough indices in that K
        # negative because we always argmax, and in mse we want smallest mse, 
        # so biggest negative MSE is smallest positive mse.
        if resample.get("mode") == "equal_to_K" and len(val_indices) == 1:
            return "negative_mse"
    return "spearman"


def architecture_config(config, index):
    safe = dict(config)
    safe.setdefault("name", f"architecture_{int(index):03d}")
    safe.setdefault("hidden_layers", [64])
    safe.setdefault("lr", 1e-3)
    safe.setdefault("weight_decay", 0.0)
    safe.setdefault("dropout", 0.0)
    safe.setdefault("batch_size", 64)
    return safe


def resolve_architecture(feature_store, y, label_spec, resamples, cache_path, args):
    cache_path = Path(cache_path)
    best_path = cache_path / "best_architecture.json"

    # If architecture is already resolved and no need to refresh, just pick it
    if best_path.is_file() and not args.refresh_architecture:
        cached = read_json(best_path)
        print(f"[architecture] using cached {best_path}")
        print(
            "[architecture] cached selection: "
            f"config={cached.get('best_config')} "
            f"best_epoch={cached.get('best_epoch')} "
            f"best_score={cached.get('best_score')} "
            f"metric={cached.get('metric_name')}",
            flush=True,
        )
        return cached


    # We need to resolve the architecture / best epoch to train
    cache_path.mkdir(parents=True, exist_ok=True)
    checkpoint_epochs = list(range(int(args.architecture_eval_every), int(args.architecture_max_epochs) + 1, int(args.architecture_eval_every)))
    if not checkpoint_epochs or checkpoint_epochs[-1] != int(args.architecture_max_epochs):
        checkpoint_epochs.append(int(args.architecture_max_epochs))

    # define the grid of all architectures
    grid = [architecture_config(config, idx) for idx, config in enumerate(DEFAULT_ARCHITECTURE_GRID)]
    scores = np.full((len(grid), len(checkpoint_epochs), len(resamples), int(args.architecture_n_seeds)), np.nan, dtype=np.float32)
    rows = []
    metric_name = architecture_selection_metric(label_spec.task_type, resamples)
    total_candidates = len(grid) * len(checkpoint_epochs) * len(resamples) * int(args.architecture_n_seeds)
    print(
        "[architecture] no cached selection; searching "
        f"architectures={len(grid)} checkpoints={len(checkpoint_epochs)} "
        f"resamples={len(resamples)} seeds={int(args.architecture_n_seeds)} "
        f"total_evaluations={total_candidates}",
        flush=True,
    )
    if metric_name == "negative_mse":
        print(
            "[architecture] equal_to_K validation has one row; "
            "selecting by negative MSE instead of correlation",
            flush=True,
        )

    # this is pretty heavy.
    # it runs PER architecture, per resample per architecture seed,
    # picks the best checkpoint, and then figures out which was the best architecture across all of those
    # and what is the best epoch. keep in mind this function gets called repeatedly, but differs in the train - validate logic.
    for arch_idx, config in enumerate(grid):

        # Train that archietcture across all resamples
        print(f"[architecture] {feature_store.model_name}: {config['name']} ({arch_idx + 1}/{len(grid)})")
        for resample_idx, resample in enumerate(resamples):
            train_idx = np.asarray(resample["train_indices"], dtype=int)
            val_idx = np.asarray(resample["val_indices"], dtype=int)

            # Assert to ensure no index leakage between train and validation sets
            assert np.isin(train_idx, val_idx).sum() == 0, "Train and validation index leakage detected"
            assert np.isin(val_idx, train_idx).sum() == 0, "validation and train index leakage detected"

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            X_train = feature_store.get(train_idx)
            normalize_eval = args.normalize_embeddings and feature_store.feature_kind == "embedding"
            if normalize_eval:
                train_mean, train_std = normalization_params(X_train)
                X_train = apply_normalization(X_train, train_mean, train_std)
            else:
                train_mean, train_std = None, None

            # as embeddings may be huge, the evaluation dataset might be too big to handle. If that is the case
            # we want to chunk it when evaluating. if not just load it all at once
            stream_val = stream_eval_in_case_of_limited_memory(feature_store, val_idx, args)
            if stream_val:
                X_val = None
                print(
                    f"[architecture] chunking emeddings (instead of loading all) total validation embeddings: rows={len(val_idx)} "
                    f"batch_size={eval_batch_size(args)}",
                    flush=True,
                )
            else:
                X_val = feature_store.get(val_idx)
                if normalize_eval:
                    X_val = apply_normalization(X_val, train_mean, train_std)
            
            # We want to test across multiple seeds to make sure we're picking the best architecture
            # by default we have 3
            for seed_idx in range(int(args.architecture_n_seeds)):
                seed = int(args.random_seed) + arch_idx * 1_000_000 + resample_idx * 1_000 + seed_idx

                # actually train, stream_val reflects whether it should be chunked or not, if that's the case
                # this function will use feature_store to get the evaluation chunk each time
                result = train_scorer_with_checkpoints(
                    X_train=X_train,
                    y_train=y[train_idx],
                    y_val=y[val_idx],
                    task_type=label_spec.task_type,
                    config=config,
                    checkpoint_epochs=checkpoint_epochs,
                    device=args.device,
                    random_seed=seed,
                    X_val=X_val,
                    feature_store=feature_store if stream_val else None,
                    val_indices=val_idx if stream_val else None,
                    normalize_eval=normalize_eval,
                    train_mean=train_mean,
                    train_std=train_std,
                    eval_max_rows=eval_batch_size(args),
                    precision_k=args.precision_k,
                    selection_metric=metric_name,
                )
                # Document what is the checkpoint score
                for ckpt_idx, epoch in enumerate(checkpoint_epochs):
                    score = result["checkpoint_scores"][ckpt_idx]
                    scores[arch_idx, ckpt_idx, resample_idx, seed_idx] = np.nan if not np.isfinite(score) else float(score)
                    rows.append(
                        {
                            "architecture_index": arch_idx,
                            "architecture_name": config["name"],
                            "epoch": int(epoch),
                            "resample_index": resample_idx,
                            "seed_index": seed_idx,
                            "seed": seed,
                            "metric_name": metric_name,
                            "score": None if not np.isfinite(score) else float(score),
                            "metrics": json.dumps(result["checkpoint_metrics"][ckpt_idx]),
                        }
                    )

    # Shape of this is architectures X checkpoints X resamples X seeds
    median_scores = np.nanmedian(scores, axis=(2, 3))
    if np.all(np.isnan(median_scores)):
        best_arch_idx, best_ckpt_idx = 0, len(checkpoint_epochs) - 1
        best_score = float("nan")
    else:
        # Figure out which is the best score, but also what is the most simple architecture with LEAST trainable parameters
        # that coul've been achieved, and earliest epoch.
        best_score = float(np.nanmax(median_scores))
        eligible = np.argwhere(median_scores >= best_score - float(args.indistinguishable_tolerance))
        best_arch_idx, best_ckpt_idx = min(
            ((int(a), int(c)) for a, c in eligible),
            key=lambda item: (
                len(grid[item[0]].get("hidden_layers", [])),
                sum(grid[item[0]].get("hidden_layers", [])),
                checkpoint_epochs[item[1]],
                -float(median_scores[item[0], item[1]]),
            ),
        )

    best_config = grid[best_arch_idx]
    best_epoch = int(checkpoint_epochs[best_ckpt_idx])
    payload = {
        "best_config": best_config,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metric_name": metric_name,
        "checkpoint_epochs": checkpoint_epochs,
        "architecture_grid": grid,
        "resample_metadata": resamples,
    }
    write_json(best_path, payload)
    pd.DataFrame(rows).to_csv(cache_path / "architecture_scores.csv", index=False)
    print(f"[architecture] selected {best_config['name']} epoch={best_epoch} {metric_name}={best_score:.4g}")
    return payload


def architecture_cache_path(label_root, feature_store, mode, k=None, split_name=None, specific_k=None):
    root = label_root / "architectures" / feature_store.model_name
    if mode == "full":
        return root / "full"
    if mode == "less_than_K":
        return root / "less_than_K" / f"K_{int(k)}"
    if mode == "equal_to_K":
        return root / "equal_to_K" / f"K_{int(k)}"
    if mode == "specific_K":
        resolved_k = int(specific_k if specific_k is not None else k)
        return root / "specific_K" / f"K_{resolved_k}"
    if mode == "random_internal":
        if split_name is None:
            raise ValueError("random_internal requires split_name")
        return root / "random_internal" / Path(split_name).stem
    raise ValueError(f"unknown architecture resolution mode: {mode}")


def resolve_for_mode(feature_store, y, label_spec, label_root, mode, args, k=None, split_name=None):
    # LOGIC:
    # first figure out what type of architecture - resolving we have.
    # "full / specific K / random interanl ..."
    # then actually resolve the best architecture given that split
    if mode == "specific_K":
        resample_k = args.specific_k if args.specific_k is not None else k
    else:
        resample_k = k

    cache_path = architecture_cache_path(
        label_root,
        feature_store,
        mode,
        k=k,
        split_name=split_name,
        specific_k=args.specific_k,
    )

    # resamples are different.
    # for random_internal, each random split gets re-splitted into 10 splits
    # for full, it's just full resampling
    # for less than K it's resampling of <= K
    # for == K it's resampling of == K
    # for equal to K it's determinstic (thus number of resamples sohould be exactly 1)
    resamples = load_resamples(label_root, mode, k=resample_k, split_name=split_name)

    print(
        "[architecture] resolving cache path: "
        f"{cache_path} mode={mode} feature={feature_store.model_name} "
        f"k={resample_k} split_name={split_name} resamples={len(resamples)}",
        flush=True,
    )
    
    # actually resolve architecture based on the validation indices
    return resolve_architecture(feature_store, y, label_spec, resamples, cache_path, args)


def run_mutation_training(df, feature_store, y, label_spec, label_root, args):
    # Architecfture resolving of random internal is relevant only
    # when we are re-splitting the random train into train-val to figure out best training architecture
    if args.architecture_resolution == "random_internal":
        raise ValueError("random_internal is only valid with --train_type random")


    nmuts = df[args.num_muts_colname].astype(int).to_numpy()
    rows_by_k = {}

    # If architecture is based on the full bootstrapped best architecture, resolve or load it
    if args.architecture_resolution == "full":
        full_arch = resolve_for_mode(feature_store, y, label_spec, label_root, "full", args)
    else:
        full_arch = None

    # For each K we are training on
    for k in range(int(args.min_muts), int(args.max_muts)):
        
        # figure ou training idices
        train_idx = np.where((nmuts >= int(args.min_muts)) & (nmuts <= k))[0].astype(int)
        if len(train_idx) == 0:
            print(f"[run] skipping K={k}; no training rows")
            continue
        
        # If architecture is full, just use it, 
        if args.architecture_resolution == "full":
            arch = full_arch
        # based on architecture resolution: 
        # 1. less than (training on <= K, picking arch based on <=K))
        # 2. equal to (training on <= K, picking arch based on == K + 1) ; also equivalent to train < k pick K +1)
        # 3. specific (training on <= K, picking arch based on specific K)
        else:
            arch = resolve_for_mode(feature_store, y, label_spec, label_root, args.architecture_resolution, args, k=k)

        # Actually train the predictor once we figured out the optimal architecture
        trained = train_predictor_from_feature_store(
            feature_store=feature_store,
            train_idx=train_idx,
            y=y,
            task_type=label_spec.task_type,
            config=arch["best_config"],
            epochs=arch["best_epoch"],
            device=args.device,
            random_seed=args.random_seed + k * 10_000,
            args=args,
        )
        
        # Now for every train mutation from K + 1 to max K, evaluate performance
        # best on the best model we trained, given the model selection we've done.
        rows = []
        for test_k in range(k + 1, int(args.max_muts) + 1):
            # train indces == k in K
            test_idx = np.where(nmuts == test_k)[0].astype(int)

            if len(test_idx) == 0:
                continue
            
            # in case of limited memory
            if stream_eval_in_case_of_limited_memory(feature_store, test_idx, args):
                print(
                    f"[run] streaming test: train_K={k} test_K={test_k} rows={len(test_idx)} "
                    f"batch_size={eval_batch_size(args)}",
                    flush=True,
                )

            # Evaluate trained model
            result = evaluate_trained_predictor_from_feature_store(
                trained,
                feature_store=feature_store,
                test_idx=test_idx,
                y=y,
                task_type=label_spec.task_type,
                device=args.device,
                args=args,
            )

            metrics = result["metrics"]
            rows.append(
                {
                    "correlation": metrics["spearman"],
                    "p_value": metrics["spearman_p_value"],
                    "test_mutations": int(test_k),
                    "train_mutations": int(k),
                    "classifier": feature_store.classifier_label,
                    "model_name": feature_store.model_name,
                    "architecture_name": arch["best_config"]["name"],
                    "best_epoch": int(arch["best_epoch"]),
                    "architecture_resolution": args.architecture_resolution,
                    "roc_auc": metrics["roc_auc"],
                    "precision": metrics["precision"],
                    "precision_at_k": metrics["precision_at_k"],
                    "precision_k": int(args.precision_k),
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "mse": metrics["mse"],
                }
            )
            del result
            gc.collect()

        rows_by_k[k] = rows
        
        out_dir = label_root / "results" / "by_mutation" / args.architecture_resolution / feature_store.model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"evaluation_on_{k}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)

        model_named = out_dir / f"{feature_store.model_name}_evaluation_train_on_{k}.csv"
        pd.DataFrame(rows).to_csv(model_named, index=False)
        
        print(f"[run] wrote {out_path}")

    return rows_by_k


def run_random_training(df, feature_store, y, label_spec, label_root, args):
    if args.architecture_resolution not in {"full", "random_internal", "specific_K"}:
        raise ValueError("random training supports --architecture_resolution full, random_internal, or specific_K")

    training_dir = label_root / "resolved_splits" / "training_splits" / "random"

    # In this case, we're using the "best resolved architecture" across the entire dataset.
    # an important emphsais, this does not mean we will leak train-test, but rather a post-hoc analysis
    # asking how would the representation behave GIVEN the best architecture possible
    # so for that we must first detect the best architecture
    if args.architecture_resolution == "full":
        full_arch = resolve_for_mode(feature_store, y, label_spec, label_root, "full", args)
    # unlike resolving internally, we can also ask what's the best architecture given a specific K
    elif args.architecture_resolution == "specific_K":
        full_arch = resolve_for_mode(feature_store, y, label_spec, label_root, "specific_K", args, k=args.specific_k)
    else:
        full_arch = None

    rows_by_size = {}
    for size in args.train_sizes:
        
        rows = []

        for split_path in sorted(training_dir.glob(f"split_{int(size)}_*.json")):
            payload = read_json(split_path)
            train_idx = np.asarray(payload["train_indices"], dtype=int)
            test_idx = np.asarray(payload["test_indices"], dtype=int)


            # Assert to ensure no index leakage between train and test sets
            assert np.isin(train_idx, test_idx).sum() == 0, "Train and test index leakage detected"
            assert np.isin(test_idx, train_idx).sum() == 0, "test and train index leakage detected"

            # In this case, for each train-test split, we must resolve the best architecture just using the 
            # train split. so for this scenarios, train validation splits were already resampled
            # from train_idx when created see PATH/scoring_cache/<class_type>/resolved_splits/validation_splits/random/ 
            if args.architecture_resolution == "random_internal":
                arch = resolve_for_mode(
                    feature_store,
                    y,
                    label_spec,
                    label_root,
                    "random_internal",
                    args,
                    split_name=split_path.name,
                )
            else:
                # In this case
                arch = full_arch

            if stream_eval_in_case_of_limited_memory(feature_store, test_idx, args):
                print(
                    f"[run] streaming random test: split={split_path.name} rows={len(test_idx)} "
                    f"batch_size={eval_batch_size(args)}",
                    flush=True,
                )

            # After figuring out what the best architecture is, we now want to actually train-test.
            result = train_final_predictor_from_feature_store(
                feature_store=feature_store,
                train_idx=train_idx,
                test_idx=test_idx,
                y=y,
                task_type=label_spec.task_type,
                config=arch["best_config"],
                epochs=arch["best_epoch"],
                device=args.device,
                random_seed=args.random_seed + int(size) * 10_000 + int(payload["iteration"]),
                args=args,
            )

            metrics = result["metrics"]

            rows.append(
                {
                    "correlation": metrics["spearman"],
                    "p_value": metrics["spearman_p_value"],
                    "test_mutations": "rest",
                    "train_mutations": int(size),
                    "classifier": feature_store.classifier_label,
                    "model_name": feature_store.model_name,
                    "split_name": split_path.stem,
                    "architecture_name": arch["best_config"]["name"],
                    "best_epoch": int(arch["best_epoch"]),
                    "architecture_resolution": args.architecture_resolution,
                    "roc_auc": metrics["roc_auc"],
                    "precision": metrics["precision"],
                    "precision_at_k": metrics["precision_at_k"],
                    "precision_k": int(args.precision_k),
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "mse": metrics["mse"],
                }
            )

        rows_by_size[int(size)] = rows
        out_dir = label_root / "results" / "random" / args.architecture_resolution / feature_store.model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"evaluation_train_size_{int(size)}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)

        print(f"[run] wrote {out_path}")

    return rows_by_size


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--cache_path")
    parser.add_argument("--onehot", action="store_true")
    parser.add_argument("--embedding", action="append", default=[])
    parser.add_argument("--regressor", action="store_true")
    parser.add_argument("--classifier", action="store_true")
    parser.add_argument("--classifier_percentile", type=float)
    parser.add_argument("--classifier_value", type=float)
    parser.add_argument("--activity_column_name", "--activity_col_name", dest="activity_column_name", required=True)
    parser.add_argument("--num_muts_colname", default="num_muts")
    parser.add_argument("--first_mutation_col")
    parser.add_argument("--last_mutation_col")
    parser.add_argument("--train_type", choices=["random", "mutation"], required=True)
    parser.add_argument("--min_muts", type=int)
    parser.add_argument("--max_muts", type=int)
    parser.add_argument("--train_sizes", type=int, nargs="*", default=[])
    parser.add_argument("--niters", type=int, default=10)
    parser.add_argument("--validation_niters", type=int, default=5)
    parser.add_argument("--validation_fraction_split", type=float, default=0.75)
    parser.add_argument("--random_internal_validation_fraction_split", type=float, nargs="*", default=[0.5, 0.75])
    parser.add_argument("--random_internal_min_val_points", type=int, default=5)
    parser.add_argument("--validation_niters_full", type=int, default=15)
    parser.add_argument("--validation_fraction_split_full", type=float, nargs="*", default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument(
        "--architecture_resolution",
        choices=["full", "less_than_K", "equal_to_K", "specific_K", "random_internal"],
        required=True,
    )
    parser.add_argument("--specific_k", type=int)
    parser.add_argument("--architecture_max_epochs", type=int, default=500)
    parser.add_argument("--architecture_eval_every", type=int, default=10)
    parser.add_argument("--architecture_n_seeds", type=int, default=3)
    parser.add_argument("--precision_k", type=int, default=100)
    parser.add_argument("--indistinguishable_tolerance", type=float, default=0.01)
    parser.add_argument("--mean_embeddings", action="store_true")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--load_all", action="store_true")
    parser.add_argument("--maximum_embeddings_to_load", type=int, default=5000)
    parser.add_argument("--cache_embedding_chunks", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh_architecture", action="store_true")
    parser.add_argument("--hard_refresh", "--hard-refresh", dest="hard_refresh", action="store_true")
    parser.add_argument("--create_missing_splits", action="store_true")
    parser.add_argument("--make_splits_only", action="store_true")
    parser.add_argument("--verbose_debug_prints", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random_seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args):
    if not args.onehot and not args.embedding:
        raise ValueError("choose --onehot and/or at least one --embedding")
    if args.regressor and (args.classifier or args.classifier_percentile is not None or args.classifier_value is not None):
        raise ValueError("--regressor cannot be combined with classifier flags")
    if args.train_type == "mutation":
        if args.min_muts is None or args.max_muts is None:
            raise ValueError("--train_type mutation requires --min_muts and --max_muts")
        if args.max_muts <= args.min_muts:
            raise ValueError("--max_muts must be greater than --min_muts")
    if args.train_type == "random" and not args.train_sizes:
        raise ValueError("--train_type random requires --train_sizes")
    if args.architecture_resolution == "specific_K" and args.specific_k is None:
        raise ValueError("--architecture_resolution specific_K requires --specific_k")
    if args.precision_k <= 0:
        raise ValueError("--precision_k must be positive")


def main():
    args = parse_args()
    validate_args(args)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    cache_path = Path(args.cache_path).expanduser().resolve() if args.cache_path else default_cache_path(dataset_path).resolve()
    df =    pd.read_csv(dataset_path)
    for required in (args.activity_column_name, args.num_muts_colname):
        if required not in df.columns:
            raise ValueError(f"missing required column {required!r}")

    label_spec = make_label_spec(df, args)
    label_root = cache_path / label_spec.cache_name
    print(f"[main] dataset={dataset_path}")
    print(f"[main] cache={label_root}")
    print(f"[main] task_type={label_spec.task_type} n_rows={len(df)}")
    if args.hard_refresh:
        hard_refresh_label_cache(label_root, cache_path)

    if split_writes_requested(args):
        create_or_update_splits(df, label_root, args)
    else:
        require_existing_splits(label_root, args)
    if args.make_splits_only:
        print("[main] make_splits_only=True; stopping after split creation")
        return

    feature_specs = []
    if args.onehot:
        feature_specs.append(("onehot", None))
    for embedding_name in args.embedding:
        feature_specs.append(("embedding", embedding_name))

    for feature_kind, embedding_name in feature_specs:
        feature_store = FeatureStore(df, dataset_path, args, feature_kind, embedding_name)
        print(f"[main] running feature={feature_store.model_name}")
        if args.train_type == "mutation":
            run_mutation_training(df, feature_store, label_spec.y, label_spec, label_root, args)
        else:
            run_random_training(df, feature_store, label_spec.y, label_spec, label_root, args)

    print("[main] done")


if __name__ == "__main__":
    main()
