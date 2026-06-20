import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import argparse

import pickle
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
# One-hot preprocessing
# ============================================================

def preprocess_one_hot(
    ohe: Union[np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Accepts:
      - [B, S, A] one-hot / categorical feature tensor
      - [B, F] already flattened features

    Returns:
      x_flat: [B, F]
      meta: dictionary describing original shape
    """
    x = to_tensor(ohe, dtype=torch.float32)

    if x.ndim == 3:
        b, s, a = x.shape
        x_flat = x.reshape(b, s * a)
        meta = {
            "input_kind": "sequence_onehot",
            "B": b,
            "S": s,
            "A": a,
            "F": s * a,
        }
        return x_flat, meta

    if x.ndim == 2:
        b, f = x.shape
        meta = {
            "input_kind": "flat_features",
            "B": b,
            "F": f,
        }
        return x, meta

    raise ValueError(
        f"One-hot input must have shape [B, S, A] or [B, F], got {tuple(x.shape)}"
    )


# ============================================================
# Baseline model
# ============================================================

class OHEMLP(nn.Module):
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
    verbose: bool = False


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(x[indices].detach(), y[indices].detach())
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


def train_one_ohe_model(
    x: torch.Tensor,
    y: torch.Tensor,
    mutation: np.ndarray,
    train_idx: np.ndarray,
    val_idx: Optional[np.ndarray],
    test_idx: np.ndarray,
    hidden_dims: Sequence[int],
    dropout: float,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, Any]:
    in_dim = x.shape[1]

    model = OHEMLP(
        in_dim=in_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_layernorm=cfg.use_layernorm,
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

    train_loader = make_loader(x, y_norm, train_idx, cfg.batch_size, shuffle=True)
    test_loader = make_loader(x, y, test_idx, cfg.batch_size, shuffle=False)

    val_loader = None
    if val_idx is not None:
        val_loader = make_loader(x, y_norm, val_idx, cfg.batch_size, shuffle=False)

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
                    f"[OHE {arch_to_str(hidden_dims):>12}] "
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
                    f"[OHE {arch_to_str(hidden_dims):>12}] "
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

    if cfg.verbose:
        print(f"Pooled test Spearman: {overall_test_spearman}")
        for metric in per_test_mutation_metrics:
            print(
                f"  mutation={metric['evaluated_test_mutation']} "
                f"n={metric['n_test_at_mutation']} "
                f"spearman={metric['test_spearman']}"
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

    if val_idx is not None:
        val_loader_orig = make_loader(x, y, val_idx, cfg.batch_size, shuffle=False)
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
# Main experiment runner
# ============================================================

def run_ohe_baseline(
    ohe: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    mutation: Union[np.ndarray, torch.Tensor],
    threshold_pairs: Sequence[Tuple[int, int]],
    validation: Optional[float] = 0.2,
    head_configs: Optional[List[Dict[str, Any]]] = None,
    train_cfg: Optional[TrainConfig] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    ohe : array-like
        [B, S, A] for position-wise one-hot
        or [B, F] if already flattened

    y : array-like, shape [B]
    mutation : array-like, shape [B]
    threshold_pairs : list of (mut_thresh_train, mut_thresh_test)
    validation : None, -1, or float in (0,1)

    For each split/config:
      - train on mutation <= mut_thresh_train
      - evaluate on mutation > mut_thresh_test
      - report one row per actual mutation count in the test set
    """
    set_seed(seed)

    x, meta = preprocess_one_hot(ohe)
    y = to_tensor(y, dtype=torch.float32).flatten()
    mutation = np.asarray(to_tensor(mutation).cpu().numpy()).astype(int)

    if len(y) != x.shape[0] or len(mutation) != x.shape[0]:
        raise ValueError("ohe, y, and mutation must agree on B")

    if train_cfg is None:
        train_cfg = TrainConfig()

    if head_configs is None:
        head_configs = [
            {"hidden_dims": [], "dropout": 0.0, "weight_decay": 0.0},
            {"hidden_dims": [128], "dropout": 0.0, "weight_decay": 1e-4},
            {"hidden_dims": [256], "dropout": 0.0, "weight_decay": 1e-4},
            {"hidden_dims": [256], "dropout": 0.2, "weight_decay": 1e-4},
            {"hidden_dims": [512], "dropout": 0.2, "weight_decay": 1e-4},
            {"hidden_dims": [256, 128], "dropout": 0.2, "weight_decay": 1e-4},
            {"hidden_dims": [512, 256], "dropout": 0.2, "weight_decay": 1e-4},
            {"hidden_dims": [512, 256, 128], "dropout": 0.3, "weight_decay": 1e-4},
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

            metrics = train_one_ohe_model(
                x=x,
                y=y,
                mutation=mutation,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                hidden_dims=hidden_dims,
                dropout=dropout,
                cfg=local_cfg,
                device=device,
            )

            for test_metric in metrics["per_test_mutation_metrics"]:
                rows.append(
                    {
                        "mut_thresh_train": mut_thresh_train,
                        "mut_thresh_test": mut_thresh_test,
                        "evaluated_test_mutation": test_metric["evaluated_test_mutation"],
                        "input_kind": meta["input_kind"],
                        "input_dim": x.shape[1],
                        "mlp_architecture": arch_to_str(hidden_dims),
                        "hidden_dims": tuple(hidden_dims),
                        "dropout": dropout,
                        "weight_decay": local_cfg.weight_decay,
                        "lr": local_cfg.lr,
                        "batch_size": local_cfg.batch_size,
                        "epochs": local_cfg.epochs,
                        "patience": local_cfg.patience,
                        "use_layernorm": local_cfg.use_layernorm,
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


# ============================================================
# Helper: best architecture per split / evaluated mutation
# ============================================================

def get_best_ohe_configs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the best configuration for each
    (mut_thresh_train, mut_thresh_test, evaluated_test_mutation)
    according to highest test_spearman.
    """
    required_cols = {
        "mut_thresh_train",
        "mut_thresh_test",
        "evaluated_test_mutation",
        "test_spearman",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    best_idx = (
        df.groupby(
            ["mut_thresh_train", "mut_thresh_test", "evaluated_test_mutation"]
        )["test_spearman"]
        .idxmax()
        .values
    )
    best_df = df.loc[best_idx].sort_values(
        ["mut_thresh_train", "mut_thresh_test", "evaluated_test_mutation"]
    ).reset_index(drop=True)
    return best_df


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OHE baseline runner")
    parser.add_argument(
        "--dataset", type=str, choices=["nmt", "pard3"], required=True,
        help="Dataset name (must be 'nmt' or 'pard3')"
    )

    # parser.add_argument(
    #     "--test_muts", type=int, required=True,
    #     help="Test set includes all samples with mutation > test_muts"
    # )

    parser.add_argument(
        "--train_muts",
        type=int,
        required=False,
        default=1,
        help="Maximum mutation threshold for training set (inclusive)",
    )

    args = parser.parse_args()
    dataset = args.dataset
    #test_muts = args.test_muts
    train_muts = args.train_muts

    data_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/data/oracle_transfer_learning"
    pickle_file_path = f"{data_path}/{dataset}_for_tl.pkl"
    output_csv_path = f"{data_path}/{dataset}_trained_on_{train_muts}_ohe_results.csv"

    with open(pickle_file_path, "rb") as f:
        all_data = pickle.load(f)
    ohe = all_data["one_hot"]
    
    y = all_data["activity"].numpy()
    mutation = all_data["mutations"].numpy().astype(int)

    #threshold_pairs = [(trmin, test_muts) for trmin in range(train_muts, test_muts)]
    threshold_pairs = [(train_muts, train_muts)]
    print(threshold_pairs)


    head_configs = [
        # linear baselines
        {"hidden_dims": [], "dropout": 0.0, "weight_decay": 0.0, "epochs": 20, "patience": 5},
        {"hidden_dims": [], "dropout": 0.0, "weight_decay": 1e-4, "epochs": 60, "patience": 10},

        # small models
        {"hidden_dims": [16], "dropout": 0.0, "weight_decay": 1e-4, "epochs": 20, "patience": 5},
        {"hidden_dims": [20], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 30, "patience": 6},
        {"hidden_dims": [32], "dropout": 0.0, "weight_decay": 1e-4, "epochs": 40, "patience": 8},
        {"hidden_dims": [50], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 50, "patience": 8},
        {"hidden_dims": [64], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 60, "patience": 10},

        # medium single-layer models
        {"hidden_dims": [100], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 60, "patience": 10},
        {"hidden_dims": [128], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 80, "patience": 12},
        {"hidden_dims": [200], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 80, "patience": 12},

        # bottleneck models
        {"hidden_dims": [64, 16], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 60, "patience": 10},
        {"hidden_dims": [100, 20], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 80, "patience": 12},
        {"hidden_dims": [128, 32], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 80, "patience": 12},
        {"hidden_dims": [200, 20], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 100, "patience": 15},

        # wider regularized models
        {"hidden_dims": [256], "dropout": 0.2, "weight_decay": 1e-4, "epochs": 80, "patience": 12},
        {"hidden_dims": [256, 64], "dropout": 0.2, "weight_decay": 1e-4, "epochs": 100, "patience": 15},
        {"hidden_dims": [256, 128], "dropout": 0.2, "weight_decay": 1e-4, "epochs": 100, "patience": 15},
        {"hidden_dims": [512, 128], "dropout": 0.3, "weight_decay": 1e-3, "epochs": 120, "patience": 20, "use_layernorm": True},

        # deeper models
        {"hidden_dims": [128, 64, 32], "dropout": 0.2, "weight_decay": 1e-4, "epochs": 100, "patience": 15},
        {"hidden_dims": [256, 128, 64], "dropout": 0.3, "weight_decay": 1e-4, "epochs": 120, "patience": 18, "use_layernorm": True},

        # even wider and deeper
        {"hidden_dims": [512, 256, 128], "dropout": 0.3, "weight_decay": 1e-3, "epochs": 150, "patience": 25, "use_layernorm": True},
        {"hidden_dims": [1024, 256], "dropout": 0.4, "weight_decay": 5e-4, "epochs": 140, "patience": 20, "use_layernorm": True},

        # deep bottleneck
        {"hidden_dims": [256, 128, 32], "dropout": 0.25, "weight_decay": 1e-4, "epochs": 120, "patience": 20, "use_layernorm": True},
        {"hidden_dims": [200, 100, 20], "dropout": 0.2, "weight_decay": 1e-4, "epochs": 120, "patience": 20, "use_layernorm": True},

        # repeated-width stack
        {"hidden_dims": [128, 128, 128], "dropout": 0.3, "weight_decay": 1e-4, "epochs": 100, "patience": 20, "use_layernorm": True},

        # small very regularized
        {"hidden_dims": [16], "dropout": 0.3, "weight_decay": 5e-4, "epochs": 40, "patience": 10},
        {"hidden_dims": [32, 8], "dropout": 0.25, "weight_decay": 1e-4, "epochs": 50, "patience": 12},
        
    # very small-data regime
    {"hidden_dims": [8], "dropout": 0.0, "weight_decay": 1e-3, "epochs": 30, "patience": 8},
    {"hidden_dims": [12], "dropout": 0.0, "weight_decay": 1e-3, "epochs": 30, "patience": 8},
    {"hidden_dims": [16, 4], "dropout": 0.1, "weight_decay": 1e-3, "epochs": 40, "patience": 10},

    # sharper bottlenecks
    {"hidden_dims": [32, 8], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 50, "patience": 12},
    {"hidden_dims": [64, 8], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 60, "patience": 12},
    {"hidden_dims": [64, 16, 4], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 70, "patience": 15},
    {"hidden_dims": [128, 16], "dropout": 0.15, "weight_decay": 1e-4, "epochs": 80, "patience": 15},
    {"hidden_dims": [256, 16], "dropout": 0.2, "weight_decay": 1e-4, "epochs": 100, "patience": 18},

    # flat-width stacks
    {"hidden_dims": [64, 64], "dropout": 0.1, "weight_decay": 1e-4, "epochs": 70, "patience": 12},
    {"hidden_dims": [128, 128], "dropout": 0.2, "weight_decay": 1e-4, "epochs": 90, "patience": 15, "use_layernorm": True},
    {"hidden_dims": [256, 256], "dropout": 0.3, "weight_decay": 1e-4, "epochs": 110, "patience": 18, "use_layernorm": True},

    # pyramids
    {"hidden_dims": [128, 64, 16], "dropout": 0.2, "weight_decay": 1e-4, "epochs": 100, "patience": 18},
    {"hidden_dims": [256, 64, 16], "dropout": 0.25, "weight_decay": 1e-4, "epochs": 120, "patience": 20, "use_layernorm": True},
    {"hidden_dims": [512, 128, 32], "dropout": 0.3, "weight_decay": 1e-3, "epochs": 140, "patience": 22, "use_layernorm": True},

    # very wide but shallow
    {"hidden_dims": [512], "dropout": 0.1, "weight_decay": 5e-4, "epochs": 90, "patience": 15},
    {"hidden_dims": [1024], "dropout": 0.2, "weight_decay": 1e-3, "epochs": 120, "patience": 20, "use_layernorm": True},
]


    train_cfg = TrainConfig(
        epochs=20,
        batch_size=32,
        lr=1e-3,
        weight_decay=0.0,
        patience=3,
        verbose=False,
    )

    df = run_ohe_baseline(
        ohe=ohe,
        y=y,
        mutation=mutation,
        threshold_pairs=threshold_pairs,
        validation=0.2,
        head_configs=head_configs,
        train_cfg=train_cfg,
        device=None,
        seed=42,
    )

    print("\nAll results:")
    print(df.head(20))

    print("\nBest config per split and evaluated test mutation:")
    df.to_csv(output_csv_path, index=False)
    best_df = get_best_ohe_configs(df)
    print(best_df)
