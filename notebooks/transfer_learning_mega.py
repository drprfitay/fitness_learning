import copy
import random
import pickle
import argparse

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x: Union[np.ndarray, torch.Tensor], dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


def safe_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho = spearmanr(y_true, y_pred).statistic
    if rho is None or np.isnan(rho):
        return np.nan
    return float(rho)


def arch_to_str(hidden_dims: Sequence[int]) -> str:
    if len(hidden_dims) == 0:
        return "linear"
    return "-".join(map(str, hidden_dims))


# ============================================================
# Split logic
# ============================================================

def make_split_masks(
    mutation: np.ndarray,
    mut_thresh_train: int,
    mut_thresh_test: int,
    validation: Optional[float] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Assumed split rule:
      train: mutation <= mut_thresh_train
      test:  mutation >  mut_thresh_test

    validation:
      - if None or -1: no validation
      - else: sample a fraction of TRAIN as validation
    """
    mutation = np.asarray(mutation)

    train_mask = mutation <= mut_thresh_train
    test_mask = mutation > mut_thresh_test

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0:
        raise ValueError(f"No training samples for mut_thresh_train={mut_thresh_train}")
    if len(test_idx) == 0:
        raise ValueError(f"No test samples for mut_thresh_test={mut_thresh_test}")

    if validation is None or validation == -1:
        return train_idx, None, test_idx

    if not (0 < validation < 1):
        raise ValueError("validation must be None, -1, or a float in (0, 1)")

    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(train_idx) * validation)))

    if n_val >= len(train_idx):
        raise ValueError(
            f"Validation split too large: train size={len(train_idx)}, requested val size={n_val}"
        )

    val_idx = np.sort(rng.choice(train_idx, size=n_val, replace=False))
    final_train_idx = np.setdiff1d(train_idx, val_idx)

    if len(final_train_idx) == 0:
        raise ValueError("Validation split removed all training points.")

    return final_train_idx, val_idx, test_idx


# ============================================================
# Layer pooling modules
# Input:  [B, S, L, D]
# Output: [B, S, D]
# ============================================================

class LayerAveragePool(nn.Module):
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return emb.mean(dim=2)


class LayerSpecificPool(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        if self.layer_idx < 0 or self.layer_idx >= emb.shape[2]:
            raise ValueError(
                f"layer_idx={self.layer_idx} is out of range for L={emb.shape[2]}"
            )
        return emb[:, :, self.layer_idx, :]


class LayerAttentionPool(nn.Module):
    """
    Learned attention over layers separately for each sequence position.

    emb: [B, S, L, D]
    output: [B, S, D]
    """
    def __init__(self, d_model: int, attn_hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        scores = self.score(emb).squeeze(-1)
        weights = torch.softmax(scores, dim=2)
        pooled = torch.sum(emb * weights.unsqueeze(-1), dim=2)
        return pooled


# ============================================================
# Sequence pooling modules
# Input:  [B, S, D]
# Output: [B, D] or [B, S*D]
# ============================================================

class SequenceGlobalPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class SequenceAveragePool(nn.Module):
    """
    To keep the 3x3 grid non-degenerate after explicit layer pooling,
    this branch preserves per-position information and flattens:
        [B, S, D] -> [B, S*D]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=1)


class SequenceAttentionPool(nn.Module):
    """
    Learned attention over sequence positions.

    x: [B, S, D]
    output: [B, D]
    """
    def __init__(self, d_model: int, attn_hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled


# ============================================================
# MLP head
# ============================================================

class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.out(x).squeeze(-1)


# ============================================================
# Full transfer model
# emb: [B, S, L, D]
# ============================================================

class TransferModel(nn.Module):
    def __init__(
        self,
        s: int,
        l: int,
        d: int,
        layer_pooling_method: str,
        sequence_pooling_method: str,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        use_layernorm: bool = False,
        layer_attn_hidden: int = 128,
        seq_attn_hidden: int = 128,
        specific_layer_idx: Optional[int] = None,
    ):
        super().__init__()

        self.layer_pooling_method = layer_pooling_method.lower()
        self.sequence_pooling_method = sequence_pooling_method.lower()

        if self.layer_pooling_method == "layer_attention":
            self.layer_pool = LayerAttentionPool(d_model=d, attn_hidden=layer_attn_hidden)
        elif self.layer_pooling_method == "layer_average":
            self.layer_pool = LayerAveragePool()
        elif self.layer_pooling_method == "layer_specific":
            if specific_layer_idx is None:
                raise ValueError("specific_layer_idx must be provided for layer_specific")
            self.layer_pool = LayerSpecificPool(layer_idx=specific_layer_idx)
        else:
            raise ValueError(f"Unknown layer_pooling_method: {layer_pooling_method}")

        if self.sequence_pooling_method == "global":
            self.sequence_pool = SequenceGlobalPool()
            in_dim = d
        elif self.sequence_pooling_method in {"average", "avg"}:
            self.sequence_pool = SequenceAveragePool()
            in_dim = s * d
        elif self.sequence_pooling_method == "attention":
            self.sequence_pool = SequenceAttentionPool(d_model=d, attn_hidden=seq_attn_hidden)
            in_dim = d
        else:
            raise ValueError(f"Unknown sequence_pooling_method: {sequence_pooling_method}")

        self.head = MLPHead(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = self.layer_pool(emb)
        x = self.sequence_pool(x)
        yhat = self.head(x)
        return yhat


# ============================================================
# Training / evaluation
# ============================================================

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 15
    use_layernorm: bool = False
    layer_attn_hidden: int = 128
    seq_attn_hidden: int = 128
    verbose: bool = False


def make_loader(
    emb: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(emb[indices].detach(), y[indices].detach())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_true = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb)
        pred = pred * y_std + y_mean

        all_preds.append(pred.cpu().numpy())
        all_true.append(yb.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    return y_true, y_pred


def summarize_test_metrics_by_mutation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_mutations: np.ndarray,
) -> List[Dict[str, Any]]:
    test_mutations = np.asarray(test_mutations)

    if len(y_true) != len(y_pred) or len(y_true) != len(test_mutations):
        raise ValueError("y_true, y_pred, and test_mutations must have the same length")

    rows = []
    for mutation_value in np.sort(np.unique(test_mutations)):
        mask = test_mutations == mutation_value
        rows.append(
            {
                "evaluated_test_mutation": int(mutation_value),
                "n_test_at_mutation": int(mask.sum()),
                "test_spearman": safe_spearmanr(y_true[mask], y_pred[mask]),
            }
        )

    return rows


def train_one_model(
    emb: torch.Tensor,
    y: torch.Tensor,
    mutation: np.ndarray,
    train_idx: np.ndarray,
    val_idx: Optional[np.ndarray],
    test_idx: np.ndarray,
    layer_pooling_method: str,
    sequence_pooling_method: str,
    hidden_dims: Sequence[int],
    dropout: float,
    cfg: TrainConfig,
    device: torch.device,
    specific_layer_idx: Optional[int] = None,
) -> Dict[str, Any]:
    _, s, l, d = emb.shape

    model = TransferModel(
        s=s,
        l=l,
        d=d,
        layer_pooling_method=layer_pooling_method,
        sequence_pooling_method=sequence_pooling_method,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_layernorm=cfg.use_layernorm,
        layer_attn_hidden=cfg.layer_attn_hidden,
        seq_attn_hidden=cfg.seq_attn_hidden,
        specific_layer_idx=specific_layer_idx,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.MSELoss()

    y_train = y[train_idx]
    y_mean = y_train.mean().item()
    y_std = y_train.std().item()
    if y_std < 1e-8:
        y_std = 1.0

    y_norm = (y - y_mean) / y_std

    train_loader = make_loader(emb, y_norm, train_idx, cfg.batch_size, shuffle=True)
    test_loader = make_loader(emb, y, test_idx, cfg.batch_size, shuffle=False)

    val_loader = None
    if val_idx is not None:
        val_loader = make_loader(emb, y_norm, val_idx, cfg.batch_size, shuffle=False)

    best_state = None
    best_epoch = -1
    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses)) if train_losses else np.nan

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    val_loss = criterion(pred, yb).item()
                    val_losses.append(val_loss)

            mean_val_loss = float(np.mean(val_losses)) if val_losses else np.inf

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if cfg.verbose:
                print(
                    f"[{layer_pooling_method:>15} x {sequence_pooling_method:>9}] "
                    f"epoch={epoch:03d} train_loss={mean_train_loss:.4f} "
                    f"val_loss={mean_val_loss:.4f}"
                )

            if epochs_without_improvement >= cfg.patience:
                break

        else:
            if mean_train_loss < best_val_loss:
                best_val_loss = mean_train_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            if cfg.verbose:
                print(
                    f"[{layer_pooling_method:>15} x {sequence_pooling_method:>9}] "
                    f"epoch={epoch:03d} train_loss={mean_train_loss:.4f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true_test, y_pred_test = predict(
        model=model,
        loader=test_loader,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
    )

    overall_test_spearman = safe_spearmanr(y_true_test, y_pred_test)
    test_mutations = np.asarray(mutation)[test_idx]
    per_test_mutation_metrics = summarize_test_metrics_by_mutation(
        y_true=y_true_test,
        y_pred=y_pred_test,
        test_mutations=test_mutations,
    )

    result = {
        "test_spearman": overall_test_spearman,
        "overall_test_spearman": overall_test_spearman,
        "per_test_mutation_metrics": per_test_mutation_metrics,
        "best_epoch": best_epoch,
        "n_train": int(len(train_idx)),
        "n_val": int(0 if val_idx is None else len(val_idx)),
        "n_test": int(len(test_idx)),
    }

    if cfg.verbose:
        print(f"Pooled test Spearman: {overall_test_spearman}")
        for metric in per_test_mutation_metrics:
            print(
                f"  mutation={metric['evaluated_test_mutation']} "
                f"n={metric['n_test_at_mutation']} "
                f"spearman={metric['test_spearman']}"
            )

    if val_idx is not None:
        val_loader_orig = make_loader(emb, y, val_idx, cfg.batch_size, shuffle=False)
        y_true_val, y_pred_val = predict(
            model=model,
            loader=val_loader_orig,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
        )
        result["val_spearman"] = safe_spearmanr(y_true_val, y_pred_val)
    else:
        result["val_spearman"] = np.nan

    return result


# ============================================================
# Base-vector delta option
# ============================================================

def apply_base_vector_delta(
    emb: torch.Tensor,
    base_vector: Optional[Union[np.ndarray, torch.Tensor]],
) -> torch.Tensor:
    """
    emb:         [B, S, L, D]
    base_vector: [1, S, L, D]
    """
    if base_vector is None:
        return emb

    base_vector = to_tensor(base_vector, dtype=torch.float32)

    if base_vector.ndim != 4:
        raise ValueError(
            f"base_vector must have 4 dims [1, S, L, D], got shape {tuple(base_vector.shape)}"
        )

    if base_vector.shape[0] != 1:
        raise ValueError(
            f"base_vector first dim must be 1, got {base_vector.shape[0]}"
        )

    if tuple(base_vector.shape[1:]) != tuple(emb.shape[1:]):
        raise ValueError(
            f"base_vector shape must be [1, S, L, D] matching emb[1:], "
            f"got base_vector={tuple(base_vector.shape)}, emb={tuple(emb.shape)}"
        )

    return emb - base_vector


# ============================================================
# Main experiment runner
# ============================================================

def run_head_transfer_learning(
    emb: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    mutation: Union[np.ndarray, torch.Tensor],
    threshold_pairs: Sequence[Tuple[int, int]],
    validation: Optional[float] = 0.2,
    layer_pooling_methods: Sequence[str] = ("layer_attention", "layer_average", "layer_specific"),
    sequence_pooling_methods: Sequence[str] = ("global", "average", "attention"),
    specific_layers: Optional[Sequence[int]] = None,
    base_vector: Optional[Union[np.ndarray, torch.Tensor]] = None,
    head_configs: Optional[List[Dict[str, Any]]] = None,
    train_cfg: Optional[TrainConfig] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    emb : array-like, shape [B, S, L, D]
    y : array-like, shape [B]
    mutation : array-like, shape [B]
    threshold_pairs : list of (mut_thresh_train, mut_thresh_test)
    validation : None, -1, or float in (0,1)

    Metrics are reported separately for each actual mutation count
    present in the test set, where test set is defined by:
        mutation > mut_thresh_test
    """
    set_seed(seed)

    emb = to_tensor(emb, dtype=torch.float32)
    y = to_tensor(y, dtype=torch.float32).flatten()
    mutation = np.asarray(to_tensor(mutation).cpu().numpy()).astype(int)

    if emb.ndim != 4:
        raise ValueError(f"emb must have shape [B, S, L, D], got {tuple(emb.shape)}")
    if y.ndim != 1:
        raise ValueError(f"y must have shape [B], got {tuple(y.shape)}")
    if len(y) != emb.shape[0] or len(mutation) != emb.shape[0]:
        raise ValueError("emb, y, and mutation must agree on B")

    emb = apply_base_vector_delta(emb, base_vector)

    _, _, L, _ = emb.shape
    if specific_layers is None:
        specific_layers = list(range(L))

    if train_cfg is None:
        train_cfg = TrainConfig()

    if head_configs is None:
        head_configs = [
            {"hidden_dims": [], "dropout": 0.0, "weight_decay": 0.0},
            {"hidden_dims": [128], "dropout": 0.0, "weight_decay": 1e-4},
            {"hidden_dims": [256], "dropout": 0.2, "weight_decay": 1e-4},
            {"hidden_dims": [256, 128], "dropout": 0.2, "weight_decay": 1e-4},
        ]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    rows = []

    for mut_thresh_train, mut_thresh_test in threshold_pairs:
        train_idx, val_idx, test_idx = make_split_masks(
            mutation=mutation,
            mut_thresh_train=mut_thresh_train,
            mut_thresh_test=mut_thresh_test,
            validation=validation,
            seed=seed,
        )

        for layer_pooling_method in layer_pooling_methods:
            layer_indices_to_run = [None]

            if layer_pooling_method == "layer_specific":
                layer_indices_to_run = list(specific_layers)

            for specific_layer_idx in layer_indices_to_run:
                for sequence_pooling_method in sequence_pooling_methods:
                    for cfg_dict in head_configs:
                        hidden_dims = cfg_dict.get("hidden_dims", [])
                        dropout = cfg_dict.get("dropout", 0.0)

                        local_cfg = copy.deepcopy(train_cfg)
                        if "weight_decay" in cfg_dict:
                            local_cfg.weight_decay = cfg_dict["weight_decay"]
                        if "lr" in cfg_dict:
                            local_cfg.lr = cfg_dict["lr"]
                        if "epochs" in cfg_dict:
                            local_cfg.epochs = cfg_dict["epochs"]
                        if "batch_size" in cfg_dict:
                            local_cfg.batch_size = cfg_dict["batch_size"]
                        if "patience" in cfg_dict:
                            local_cfg.patience = cfg_dict["patience"]
                        if "use_layernorm" in cfg_dict:
                            local_cfg.use_layernorm = cfg_dict["use_layernorm"]
                        if "layer_attn_hidden" in cfg_dict:
                            local_cfg.layer_attn_hidden = cfg_dict["layer_attn_hidden"]
                        if "seq_attn_hidden" in cfg_dict:
                            local_cfg.seq_attn_hidden = cfg_dict["seq_attn_hidden"]

                        metrics = train_one_model(
                            emb=emb,
                            y=y,
                            mutation=mutation,
                            train_idx=train_idx,
                            val_idx=val_idx,
                            test_idx=test_idx,
                            layer_pooling_method=layer_pooling_method,
                            sequence_pooling_method=sequence_pooling_method,
                            hidden_dims=hidden_dims,
                            dropout=dropout,
                            cfg=local_cfg,
                            device=device,
                            specific_layer_idx=specific_layer_idx,
                        )

                        pooling_name = f"{layer_pooling_method} x {sequence_pooling_method}"
                        if layer_pooling_method == "layer_specific":
                            pooling_name += f" (layer={specific_layer_idx})"

                        for test_metric in metrics["per_test_mutation_metrics"]:
                            rows.append(
                                {
                                    "mut_thresh_train": mut_thresh_train,
                                    "mut_thresh_test": mut_thresh_test,
                                    "evaluated_test_mutation": test_metric["evaluated_test_mutation"],
                                    "layer_pooling_method": layer_pooling_method,
                                    "sequence_pooling_method": sequence_pooling_method,
                                    "pooling_method": pooling_name,
                                    "specific_layer_idx": (
                                        np.nan if specific_layer_idx is None else specific_layer_idx
                                    ),
                                    "mlp_architecture": arch_to_str(hidden_dims),
                                    "hidden_dims": tuple(hidden_dims),
                                    "dropout": dropout,
                                    "weight_decay": local_cfg.weight_decay,
                                    "lr": local_cfg.lr,
                                    "batch_size": local_cfg.batch_size,
                                    "epochs": local_cfg.epochs,
                                    "patience": local_cfg.patience,
                                    "use_layernorm": local_cfg.use_layernorm,
                                    "layer_attn_hidden": (
                                        local_cfg.layer_attn_hidden
                                        if layer_pooling_method == "layer_attention"
                                        else np.nan
                                    ),
                                    "seq_attn_hidden": (
                                        local_cfg.seq_attn_hidden
                                        if sequence_pooling_method == "attention"
                                        else np.nan
                                    ),
                                    "used_base_vector_delta": base_vector is not None,
                                    "validation_fraction": validation,
                                    "used_validation": val_idx is not None,
                                    "n_train": metrics["n_train"],
                                    "n_val": metrics["n_val"],
                                    "n_test": metrics["n_test"],
                                    "n_test_at_mutation": test_metric["n_test_at_mutation"],
                                    "best_epoch": metrics["best_epoch"],
                                    "val_spearman": metrics["val_spearman"],
                                    "overall_test_spearman": metrics["overall_test_spearman"],
                                    "test_spearman": test_metric["test_spearman"],
                                }
                            )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["mut_thresh_train", "mut_thresh_test", "evaluated_test_mutation", "test_spearman"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning TL heads runner")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nmt", "pard3"],
        required=True,
        help="Dataset name (must be 'nmt' or 'pard3')",
    )
    # parser.add_argument(
    #     "--test_muts",
    #     type=int,
    #     required=True,
    #     help="Test set includes samples with mutation > test_muts",
    # )
    parser.add_argument(
        "--train_muts",
        type=int,
        required=False,
        default=1,
        help="Maximum mutation threshold for training set (inclusive)",
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        default=False,
        help="Whether to use delta embeddings (base_vector subtraction)",
    )

    args = parser.parse_args()

    delta = args.delta
    dataset = args.dataset
    #test_muts = args.test_muts
    train_muts = args.train_muts
    prefix = "delta" if delta else "regular"

    data_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/data/oracle_transfer_learning"
    pickle_file_path = f"{data_path}/{dataset}_for_tl.pkl"
    output_csv_path = f"{data_path}/{dataset}_trained_on_{train_muts}_{prefix}_tl_results.csv"

    with open(pickle_file_path, "rb") as f:
        all_data = pickle.load(f)

    emb = all_data["repr"]
    y = all_data["activity"].numpy()
    mutation = all_data["mutations"].numpy().astype(int)

    base_vector = None
    if delta:
        wt_mask = mutation == 0
        if wt_mask.sum() == 0:
            raise ValueError("Cannot use --delta because no mutation==0 samples were found")

        if isinstance(emb, torch.Tensor):
            wt_mask_tensor = torch.as_tensor(wt_mask, dtype=torch.bool, device=emb.device)
            base_vector = emb[wt_mask_tensor].mean(dim=0, keepdim=True)
        else:
            base_vector = emb[wt_mask].mean(axis=0, keepdims=True)

    threshold_pairs = [(train_muts, train_muts)]
    print(threshold_pairs)

    head_configs = [
        {"hidden_dims": [128], "dropout": 0.1, "weight_decay": 1e-4},
    ]

    train_cfg = TrainConfig(
        epochs=70,
        batch_size=32,
        lr=1e-3,
        weight_decay=0.0,
        patience=10,
        verbose=True,
    )

    df = run_head_transfer_learning(
        emb=emb,
        y=y,
        mutation=mutation,
        threshold_pairs=threshold_pairs,
        validation=0.2,
        layer_pooling_methods=("layer_attention", "layer_average", "layer_specific"),
        sequence_pooling_methods=("global", "average", "attention"),
        specific_layers=[0, 1, 2, 3, 4, 5],
        base_vector=base_vector,
        head_configs=head_configs,
        train_cfg=train_cfg,
        device=None,
        seed=42,
    )

    df.to_csv(output_csv_path, index=False)
    print(df.head(20))
