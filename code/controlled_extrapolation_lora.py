#!/usr/bin/env python3
"""Controlled-extrapolation LoRA fine-tuning for mutation-order splits."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def find_project_paths():
    script_path = Path(__file__).resolve()
    for parent in [script_path.parent] + list(script_path.parents):
        if (parent / "plm_base.py").exists() and (parent.parent / "models/esm2").exists():
            return parent, parent.parent
        if (parent / "code/plm_base.py").exists() and (parent / "models/esm2").exists():
            return parent / "code", parent
    raise RuntimeError("could not find fitness_learning code/ and models/esm2 directories from %s" % script_path)


CODE_DIR, ROOT_DIR = find_project_paths()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from new_trainer import configure_trainable_parameters, resolve_token_ids
from plm_base import plmEmbeddingModel, plm_init
from sequence_embedding_scoring_analysis import MLPScorer, evaluate_predictions


RESULT_COLUMNS = [
    "roc",
    "clf_type",
    "top_100_pct",
    "test_mutations",
    "train_mutations",
    "dataset",
    "model_name",
    "correlation",
]

MODEL_ALIASES = {
    "esm_8m": "esm2_t6_8M_UR50D",
    "esm8m": "esm2_t6_8M_UR50D",
    "esm_35m": "esm2_t12_35M_UR50D",
    "esm35m": "esm2_t12_35M_UR50D",
    "esm_150m": "esm2_t30_150M_UR50D",
    "esm150m": "esm2_t30_150M_UR50D",
    "esm_650m": "esm2_t33_650M_UR50D",
    "esm650m": "esm2_t33_650M_UR50D",
    "esm_3b": "esm2_t36_3B_UR50D",
    "esm3b": "esm2_t36_3B_UR50D",
}

DATASET_CONFIGS = {
    "gfp": {
        "path": ROOT_DIR / "data/configuration/fixed_unique_gfp_sequence_dataset_full_seq.csv",
        "sequence_column": "full_seq",
        "num_muts_column": "num_muts",
        "label_column": "inactive",
        "task_type": "classification",
    },
    "lov": {
        "path": ROOT_DIR / "notebooks/data/raw_csvs/LOV.csv",
        "sequence_column": "full_seq",
        "num_muts_column": "num_muts",
        "label_column": "activity",
        "task_type": "regression",
    },
    "pard3": {
        "path": ROOT_DIR / "notebooks/data/raw_csvs/PARD3.csv",
        "sequence_column": "full_seq",
        "num_muts_column": "num_muts",
        "label_column": "activity",
        "task_type": "regression",
    },
    "gcn4": {
        "path": ROOT_DIR / "notebooks/data/raw_csvs/GCN4_YEAST_Staller_2018_for_llm_embedding.csv",
        "sequence_column": "full_seq",
        "num_muts_column": "num_muts",
        "label_column": "activity",
        "task_type": "regression",
    },
}


@dataclass
class DatasetSpec:
    name: str
    path: Path
    sequence_column: str
    num_muts_column: str
    label_column: str
    task_type: str


class SequenceDataset(Dataset):
    def __init__(self, encoded, labels):
        self.encoded = encoded
        self.labels = torch.as_tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(self.encoded, torch.Tensor):
            input_ids = self.encoded[idx]
        else:
            input_ids = self.encoded[idx]
        return torch.as_tensor(input_ids, dtype=torch.long), self.labels[idx]


class LoraSupervisedModel(nn.Module):
    def __init__(self, backbone, head, pooling_positions=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pooling_positions = pooling_positions

    def forward(self, input_ids, attention_mask=None):
        try:
            hidden = self.backbone(input_ids, attention_mask=attention_mask)
        except TypeError:
            hidden = self.backbone(input_ids)
        pooled = self.pool_hidden(hidden, input_ids, attention_mask)
        return self.head(pooled)

    def pool_hidden(self, hidden, input_ids, attention_mask):
        if self.pooling_positions:
            valid_positions = [pos for pos in self.pooling_positions if pos < hidden.shape[1]]
            if valid_positions:
                emb = hidden[:, torch.as_tensor(valid_positions, device=hidden.device), :]
                emb = torch.nn.functional.normalize(emb, dim=1).mean(dim=1)
                return torch.nn.functional.normalize(emb, dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()
        if hidden.shape[1] > 2:
            attention_mask = attention_mask.clone()
            attention_mask[:, 0] = False
            attention_mask[:, -1] = False
        denom = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(1)
        emb = (hidden * attention_mask.unsqueeze(2)).sum(dim=1) / denom
        return torch.nn.functional.normalize(emb, dim=1)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device):
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def infer_dataset_spec(dataset):
    dataset_path = Path(dataset).expanduser()
    if dataset in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset]
        return DatasetSpec(
            name=dataset,
            path=Path(config["path"]),
            sequence_column=config["sequence_column"],
            num_muts_column=config["num_muts_column"],
            label_column=config["label_column"],
            task_type=config["task_type"],
        )
    if dataset_path.exists():
        name = dataset_path.stem
        df_head = pd.read_csv(dataset_path, nrows=5)
        sequence_column = "full_seq" if "full_seq" in df_head.columns else "sequence"
        num_muts_column = "num_muts" if "num_muts" in df_head.columns else "num_of_muts"
        if "inactive" in df_head.columns:
            label_column = "inactive"
            task_type = "classification"
        elif "activity" in df_head.columns:
            label_column = "activity"
            task_type = "regression"
        elif "binned_activity" in df_head.columns:
            label_column = "binned_activity"
            task_type = "classification"
        else:
            raise ValueError("could not infer label column; expected inactive, activity, or binned_activity")
        return DatasetSpec(name, dataset_path, sequence_column, num_muts_column, label_column, task_type)
    raise ValueError("unknown dataset %r; use one of %s or pass a CSV path" % (dataset, ", ".join(sorted(DATASET_CONFIGS))))


def load_prepare_dataset(dataset):
    spec = infer_dataset_spec(dataset)
    print("Loading dataset %s" % spec.name)
    df = pd.read_csv(spec.path)
    for column in (spec.sequence_column, spec.num_muts_column, spec.label_column):
        if column not in df.columns:
            raise ValueError("dataset %s is missing required column %r" % (spec.path, column))
    df = df.reset_index(drop=True)
    labels = make_labels(df, spec)
    nmuts = df[spec.num_muts_column].astype(int).to_numpy()
    print("Task: %s label_column=%s sequence_column=%s num_muts_column=%s" % (
        spec.task_type, spec.label_column, spec.sequence_column, spec.num_muts_column
    ))
    return spec, df, labels, nmuts


def make_labels(df, spec):
    values = df[spec.label_column]
    if spec.task_type == "regression":
        return values.astype(float).to_numpy()
    labels = pd.Categorical(values).codes.astype(int)
    if np.any(labels < 0):
        raise ValueError("classification labels contain NaN values")
    return labels


def build_controlled_indices(nmuts, train_mutations):
    train_idx = np.where(nmuts <= int(train_mutations))[0]
    test_orders = sorted(int(k) for k in np.unique(nmuts) if int(k) > int(train_mutations))
    return train_idx, test_orders


def infer_pooling_positions(df, spec):
    positions = []
    for column in df.columns:
        if column in {spec.sequence_column, spec.num_muts_column, spec.label_column}:
            continue
        if len(column) >= 2 and column[0].isalpha() and column[1:].isdigit():
            positions.append(int(column[1:]))
    return positions or None


def load_pretokenized(tokenized_path, n_rows):
    if tokenized_path is None:
        return None
    obj = torch.load(tokenized_path, map_location="cpu")
    indices = None
    if isinstance(obj, dict):
        if "input_ids" in obj:
            encoded = obj["input_ids"]
        elif "encoded_tensor" in obj:
            encoded = obj["encoded_tensor"]
        else:
            raise ValueError("tokenized dict must contain input_ids or encoded_tensor")
        indices = obj.get("indices")
    else:
        encoded = obj
    if indices is not None:
        indices = np.asarray(indices, dtype=int)
        if len(indices) != len(encoded):
            raise ValueError("tokenized indices and input_ids have different lengths")
        aligned = [None] * n_rows
        for local_idx, global_idx in enumerate(indices):
            aligned[int(global_idx)] = encoded[local_idx]
        if any(item is None for item in aligned):
            raise ValueError("tokenized indices do not cover every dataset row")
        encoded = aligned
    if len(encoded) != n_rows:
        raise ValueError("tokenized input length %d does not match dataset rows %d" % (len(encoded), n_rows))
    print("Loaded pre-tokenized inputs from %s" % tokenized_path)
    return encoded


def infer_hidden_dim(backbone):
    plm = backbone.plm
    if hasattr(plm, "embed_tokens"):
        return int(plm.embed_tokens.weight.shape[1])
    if hasattr(plm, "get_input_embeddings") and plm.get_input_embeddings() is not None:
        return int(plm.get_input_embeddings().weight.shape[1])
    if hasattr(plm, "transformer") and hasattr(plm.transformer, "wte"):
        return int(plm.transformer.wte.weight.shape[1])
    if hasattr(plm, "embeddings") and hasattr(plm.embeddings, "word_embeddings"):
        return int(plm.embeddings.word_embeddings.weight.shape[1])
    raise ValueError("could not infer PLM hidden dimension for supervised head")


def tokenize_sequences(df, spec, model, model_name):
    cache_path = spec.path.with_suffix("")
    cache_path = Path(str(cache_path) + "_cache") / "misc"
    tokenized_sequences_filename = "%s_encoded_sequences.pt" % model_name
    cached_path = cache_path / tokenized_sequences_filename
    if cached_path.exists():
        print("[INFO] Loading cached file %s" % tokenized_sequences_filename)
        return torch.load(cached_path, map_location="cpu")
    print("[INFO] Tokenizing sequences, this may take a while")
    encoded = [torch.as_tensor(model.encode(seq), dtype=torch.long) for seq in df[spec.sequence_column].to_list()]
    lengths = {int(x.numel()) for x in encoded}
    if len(lengths) == 1:
        encoded = torch.stack(encoded, dim=0)
    cache_path.mkdir(parents=True, exist_ok=True)
    torch.save(encoded, cached_path)
    return encoded


def collate_batch(batch, pad_id):
    sequences, labels = zip(*batch)
    max_len = max(seq.numel() for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, : seq.numel()] = seq
        attention_mask[i, : seq.numel()] = 1
    return input_ids, attention_mask, torch.stack([torch.as_tensor(label) for label in labels])


def subset_encoded(encoded, indices):
    indices = np.asarray(indices, dtype=int)
    if isinstance(encoded, torch.Tensor):
        return encoded[torch.as_tensor(indices, dtype=torch.long)]
    return [encoded[int(i)] for i in indices]


def initialize_plm_lora_head(model_name, task_type, output_dim, device, lora_rank, lora_alpha, lora_dropout, pooling_positions):
    resolved_model_name = resolve_model_name(model_name)
    print("Model: %s" % resolved_model_name)
    plm_init(str(ROOT_DIR))
    backbone = plmEmbeddingModel(plm_name=resolved_model_name, emb_only=True, device=device).to(device)
    configure_trainable_parameters(backbone.plm, "lora", lora_r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    hidden_dim = infer_hidden_dim(backbone)
    head = MLPScorer(input_dim=int(hidden_dim), output_dim=int(output_dim), hidden_layers=[64], dropout=0.0).to(device)
    model = LoraSupervisedModel(backbone, head, pooling_positions=pooling_positions).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Trainable parameters: %d / %d" % (trainable_params, total_params))
    return model, backbone


def train(model, encoded, labels, train_idx, task_type, token_ids, args, device):
    train_dataset = SequenceDataset(subset_encoded(encoded, train_idx), labels[train_idx])
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, int(token_ids["pad"])),
    )
    if len(loader) == 0:
        raise ValueError("empty training set")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    print("Fitting LoRA...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for input_ids, attention_mask, y in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids, attention_mask=attention_mask)
            if task_type == "classification":
                loss = loss_fn(output, y.long().to(device))
            else:
                loss = loss_fn(output.reshape(-1), y.float().to(device).reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print("Epoch %d/%d loss=%.6f" % (epoch, args.epochs, float(np.mean(losses))))
    return model


@torch.no_grad()
def predict(model, encoded, labels, indices, task_type, token_ids, eval_batch_size, device):
    dataset = SequenceDataset(subset_encoded(encoded, indices), labels[indices])
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, int(token_ids["pad"])),
    )
    predictions = []
    model.eval()
    for input_ids, attention_mask, _y in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids, attention_mask=attention_mask)
        if task_type == "classification":
            predictions.append(torch.softmax(output, dim=1).detach().cpu().numpy())
        else:
            predictions.append(output.reshape(-1).detach().cpu().numpy())
    if not predictions:
        return np.asarray([])
    return np.concatenate(predictions, axis=0)


def evaluate_one_order(model, encoded, labels, nmuts, test_order, spec, args, token_ids, device):
    test_idx = np.where(nmuts == int(test_order))[0]
    print("Evaluating mutation order %d, N=%d" % (int(test_order), len(test_idx)))
    if len(test_idx) == 0:
        return None
    y_true = labels[test_idx]
    y_pred = predict(model, encoded, labels, test_idx, spec.task_type, token_ids, args.eval_batch_size, device)
    if spec.task_type == "classification" and len(np.unique(y_true)) < 2:
        print("[WARNING] Mutation order %d contains only one class; ROC AUC is NA" % int(test_order))
    metrics = evaluate_predictions(y_true, y_pred, spec.task_type, precision_k=100)
    if spec.task_type == "classification":
        print("ROC AUC: %s" % metrics["roc_auc"])
        return {
            "roc": metrics["roc_auc"],
            "clf_type": spec.task_type,
            "top_100_pct": metrics["precision_at_k"],
            "test_mutations": int(test_order),
            "train_mutations": int(args.train_mutations),
            "dataset": spec.name,
            "model_name": args.model_name,
            "correlation": np.nan,
        }
    print("Spearman correlation: %s" % metrics["spearman"])
    return {
        "roc": np.nan,
        "clf_type": spec.task_type,
        "top_100_pct": np.nan,
        "test_mutations": int(test_order),
        "train_mutations": int(args.train_mutations),
        "dataset": spec.name,
        "model_name": args.model_name,
        "correlation": metrics["spearman"],
    }


def save_result_incremental(row, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row], columns=RESULT_COLUMNS)
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        for column in RESULT_COLUMNS:
            if column not in old_df.columns:
                old_df[column] = np.nan
        out_df = pd.concat([old_df[RESULT_COLUMNS], new_df], ignore_index=True)
    else:
        out_df = new_df
    out_df = out_df.drop_duplicates(
        subset=["dataset", "model_name", "train_mutations", "test_mutations", "clf_type"],
        keep="last",
    )
    out_df.to_csv(output_path, index=False, columns=RESULT_COLUMNS)
    print("Saved results to %s" % output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_mutations", type=int, required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenized_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print("Using device: %s" % device)
    spec, df, labels, nmuts = load_prepare_dataset(args.dataset)
    train_idx, test_orders = build_controlled_indices(nmuts, args.train_mutations)
    print("Training orders: <= %d" % int(args.train_mutations))
    print("N train: %d" % len(train_idx))
    print("Held-out orders: %s" % (", ".join(map(str, test_orders)) if test_orders else "<none>"))

    pooling_positions = infer_pooling_positions(df, spec)
    if pooling_positions is not None:
        print("Pooling designed positions: %s" % ", ".join(map(str, pooling_positions)))

    output_dim = int(np.max(labels)) + 1 if spec.task_type == "classification" else 1
    model, backbone = initialize_plm_lora_head(
        args.model_name,
        spec.task_type,
        output_dim,
        device,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        pooling_positions,
    )
    token_ids = resolve_token_ids(backbone.tokenizer)
    if token_ids["pad"] is None:
        token_ids["pad"] = 0

    encoded = load_pretokenized(args.tokenized_path, len(df))
    if encoded is None:
        encoded = tokenize_sequences(df, spec, backbone, resolve_model_name(args.model_name))

    model = train(model, encoded, labels, train_idx, spec.task_type, token_ids, args, device)
    for test_order in test_orders:
        row = evaluate_one_order(model, encoded, labels, nmuts, test_order, spec, args, token_ids, device)
        if row is not None:
            save_result_incremental(row, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
