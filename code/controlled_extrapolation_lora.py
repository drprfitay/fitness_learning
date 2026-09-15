#!/usr/bin/env python3
"""Controlled-extrapolation LoRA fine-tuning for mutation-order splits."""

from __future__ import annotations

import argparse
import math
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

from new_trainer import resolve_token_ids
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

PREDICTION_COLUMNS = [
    "dataset",
    "model_name",
    "train_mutations",
    "split",
    "mutation_order",
    "row_index",
    "target",
    "prediction",
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

LORA_TARGET_ALIASES = {
    "query": ["q_proj", "query"],
    "key": ["k_proj", "key"],
    "value": ["v_proj", "value"],
    "output": ["out_proj", "dense"],
}

LORA_SCOPE_TARGETS = {
    "qv": ["query", "value"],
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


def transformer_backbone_roots(model):
    roots = []
    if hasattr(model, "layers"):
        roots.append(("layers", model.layers))
    if hasattr(model, "encoder"):
        encoder = model.encoder
        if hasattr(encoder, "layer"):
            roots.append(("encoder.layer", encoder.layer))
        if hasattr(encoder, "block"):
            roots.append(("encoder.block", encoder.block))
        if hasattr(encoder, "layers"):
            roots.append(("encoder.layers", encoder.layers))
    if hasattr(model, "transformer"):
        transformer = model.transformer
        if hasattr(transformer, "h"):
            roots.append(("transformer.h", transformer.h))
        if hasattr(transformer, "layers"):
            roots.append(("transformer.layers", transformer.layers))
    if not roots:
        raise ValueError("could not identify transformer backbone layers for LoRA targeting")
    return roots


def iter_backbone_linear_modules(model):
    for root_name, root_module in transformer_backbone_roots(model):
        for name, module in root_module.named_modules():
            if isinstance(module, nn.Linear):
                full_name = "%s.%s" % (root_name, name) if name else root_name
                yield full_name, module


def resolve_named_lora_targets(model, requested_targets):
    requested_targets = [str(target) for target in requested_targets]
    linear_module_names = [name for name, _module in iter_backbone_linear_modules(model)]
    linear_leaf_names = {name.split(".")[-1] for name in linear_module_names}
    resolved_leaf_names = []
    for requested in requested_targets:
        candidates = LORA_TARGET_ALIASES.get(requested, [requested])
        matched = [candidate for candidate in candidates if candidate in linear_leaf_names]
        if not matched:
            raise ValueError(
                "could not resolve LoRA target module %r under transformer backbone. Available Linear leaf names include: %s" %
                (requested, ", ".join(sorted(linear_leaf_names)[:50]))
            )
        resolved_leaf_names.append(matched[0])
    return sorted(name for name in linear_module_names if name.split(".")[-1] in set(resolved_leaf_names))


def resolve_lora_target_modules(model, args):
    if args.lora_scope == "all_linear":
        return sorted(name for name, _module in iter_backbone_linear_modules(model))
    if args.lora_scope == "qv":
        requested_targets = args.lora_target_modules or LORA_SCOPE_TARGETS["qv"]
        return resolve_named_lora_targets(model, requested_targets)
    raise ValueError("unsupported LoRA scope: %s" % args.lora_scope)


def replace_target_linear_with_lora(module, target_module_names, r, alpha, dropout, prefix=""):
    import loralib as lora

    target_module_names = set(target_module_names)
    matched = []
    for name, child in list(module.named_children()):
        full_name = "%s.%s" % (prefix, name) if prefix else name
        if isinstance(child, nn.Linear) and full_name in target_module_names:
            replacement = lora.Linear(
                child.in_features,
                child.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias=child.bias is not None,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            replacement.train(child.training)
            setattr(module, name, replacement)
            matched.append(full_name)
        else:
            matched.extend(
                replace_target_linear_with_lora(
                    child,
                    target_module_names,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    prefix=full_name,
                )
            )
    return matched


def summarize_lora_modules(matched_modules):
    groups = {
        "attention_query": [],
        "attention_key": [],
        "attention_value": [],
        "attention_output": [],
        "ffn_intermediate": [],
        "ffn_output": [],
        "other": [],
    }
    for name in matched_modules:
        leaf = name.split(".")[-1]
        if leaf in {"q_proj", "query"}:
            groups["attention_query"].append(name)
        elif leaf in {"k_proj", "key"}:
            groups["attention_key"].append(name)
        elif leaf in {"v_proj", "value"}:
            groups["attention_value"].append(name)
        elif leaf in {"out_proj"} or name.endswith(".self_attn.out_proj"):
            groups["attention_output"].append(name)
        elif leaf in {"fc1", "intermediate", "dense_h_to_4h"}:
            groups["ffn_intermediate"].append(name)
        elif leaf in {"fc2", "output", "dense_4h_to_h"}:
            groups["ffn_output"].append(name)
        else:
            groups["other"].append(name)
    return groups


def print_lora_module_summary(lora_scope, lora_rank, lora_alpha, lora_dropout, matched_modules, total_params, trainable_params):
    trainable_pct = 100.0 * float(trainable_params) / float(total_params) if total_params else 0.0
    groups = summarize_lora_modules(matched_modules)
    print("LoRA model summary:")
    print("  lora_scope: %s" % lora_scope)
    print("  lora_rank: %d" % lora_rank)
    print("  lora_alpha: %d" % lora_alpha)
    print("  lora_dropout: %.4g" % lora_dropout)
    print("  total_parameters: %d" % total_params)
    print("  trainable_parameters: %d" % trainable_params)
    print("  trainable_pct: %.4f" % trainable_pct)
    print("LoRA modules by layer type:")
    print("  Attention:")
    print("    query: %d" % len(groups["attention_query"]))
    print("    key: %d" % len(groups["attention_key"]))
    print("    value: %d" % len(groups["attention_value"]))
    print("    output projection: %d" % len(groups["attention_output"]))
    print("  FFN:")
    print("    intermediate projection: %d" % len(groups["ffn_intermediate"]))
    print("    output projection: %d" % len(groups["ffn_output"]))
    if groups["other"]:
        print("  Other linear: %d" % len(groups["other"]))
    print("Exact modules receiving LoRA:")
    for module_name in matched_modules:
        print("  %s" % module_name)


def sanity_check_trainable_parameters(model, matched_modules):
    expected_lora_prefixes = tuple("backbone.plm.%s." % name for name in matched_modules)
    unexpected_trainable = []
    trainable_lora = []
    frozen_lora = []
    head_trainable = []
    head_frozen = []
    for name, param in model.named_parameters():
        if name.startswith("head."):
            if param.requires_grad:
                head_trainable.append(name)
            else:
                head_frozen.append(name)
        elif "lora_" in name:
            if param.requires_grad:
                trainable_lora.append(name)
            else:
                frozen_lora.append(name)
        elif param.requires_grad:
            unexpected_trainable.append(name)

    missing_expected_lora = [
        prefix for prefix in expected_lora_prefixes
        if not any(name.startswith(prefix) and "lora_" in name for name in trainable_lora)
    ]
    if unexpected_trainable:
        raise RuntimeError("unexpected base-model trainable parameters: %s" % ", ".join(unexpected_trainable[:20]))
    if frozen_lora:
        raise RuntimeError("LoRA parameters unexpectedly frozen: %s" % ", ".join(frozen_lora[:20]))
    if head_frozen:
        raise RuntimeError("prediction-head parameters unexpectedly frozen: %s" % ", ".join(head_frozen[:20]))
    if missing_expected_lora:
        raise RuntimeError("missing trainable LoRA adapters for modules: %s" % ", ".join(missing_expected_lora[:20]))
    if not trainable_lora:
        raise RuntimeError("no trainable LoRA parameters found")
    if not head_trainable:
        raise RuntimeError("no trainable prediction-head parameters found")
    print("LoRA sanity check passed: base pLM frozen, LoRA adapters trainable, prediction head trainable")


def configure_lora_parameters(model, args):
    resolved_targets = resolve_lora_target_modules(model, args)
    matched_modules = replace_target_linear_with_lora(
        model,
        resolved_targets,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if not matched_modules:
        raise ValueError("no modules were replaced with LoRA for targets: %s" % ", ".join(resolved_targets))
    for _name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    return resolved_targets, matched_modules


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


def supports_bf16(device):
    return device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def cosine_warmup_scheduler(optimizer, total_steps, warmup_ratio):
    total_steps = max(int(total_steps), 1)
    warmup_steps = int(total_steps * float(warmup_ratio))

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max(float(current_step - warmup_steps) / float(decay_steps), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), warmup_steps


def trainable_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if "lora_" in name or name.startswith("head.")
    }


def initialization_equivalence_batch(encoded, indices, labels, token_ids, max_examples):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return None
    indices = indices[: int(max_examples)]
    dataset = SequenceDataset(subset_encoded(encoded, indices), labels[indices])
    return collate_batch([dataset[i] for i in range(len(dataset))], int(token_ids["pad"]))


@torch.no_grad()
def check_lora_initialization_equivalence(backbone, encoded, labels, indices, token_ids, device, max_examples):
    batch = initialization_equivalence_batch(encoded, indices, labels, token_ids, max_examples)
    if batch is None:
        print("[WARNING] Skipping LoRA initialization equivalence check because no examples were available")
        return None
    was_training = backbone.training
    backbone.eval()
    input_ids, attention_mask, _y = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    before = backbone(input_ids, attention_mask=attention_mask).detach().float().cpu()
    if was_training:
        backbone.train()
    return input_ids, attention_mask, before, was_training


@torch.no_grad()
def finish_lora_initialization_equivalence(backbone, equivalence_state):
    if equivalence_state is None:
        return
    input_ids, attention_mask, before, was_training = equivalence_state
    backbone.eval()
    after = backbone(input_ids, attention_mask=attention_mask).detach().float().cpu()
    diff = (before - after).abs()
    print("LoRA initialization equivalence check:")
    print("  max_absolute_difference: %.8g" % float(diff.max().item()))
    print("  mean_absolute_difference: %.8g" % float(diff.mean().item()))
    if was_training:
        backbone.train()


def initialize_plm_lora_head(
    model_name,
    task_type,
    output_dim,
    device,
    args,
    pooling_positions,
    encoded=None,
    labels=None,
    diagnostic_indices=None,
    report_init_equivalence=False,
):
    resolved_model_name = resolve_model_name(model_name)
    print("Model: %s" % resolved_model_name)
    plm_init(str(ROOT_DIR))
    backbone = plmEmbeddingModel(plm_name=resolved_model_name, emb_only=True, device=device).to(device)
    token_ids = resolve_token_ids(backbone.tokenizer)
    if token_ids["pad"] is None:
        token_ids["pad"] = 0
    equivalence_state = None
    if report_init_equivalence and encoded is not None and labels is not None and diagnostic_indices is not None:
        equivalence_state = check_lora_initialization_equivalence(
            backbone,
            encoded,
            labels,
            diagnostic_indices,
            token_ids,
            device,
            max_examples=min(args.eval_batch_size, 8),
        )
    _resolved_targets, matched_modules = configure_lora_parameters(backbone.plm, args)
    finish_lora_initialization_equivalence(backbone, equivalence_state)
    hidden_dim = infer_hidden_dim(backbone)
    head = MLPScorer(input_dim=int(hidden_dim), output_dim=int(output_dim), hidden_layers=[64], dropout=0.0).to(device)
    for param in head.parameters():
        param.requires_grad = True
    model = LoraSupervisedModel(backbone, head, pooling_positions=pooling_positions).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print_lora_module_summary(
        args.lora_scope,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        matched_modules,
        total_params,
        trainable_params,
    )
    sanity_check_trainable_parameters(model, matched_modules)
    return model, backbone, token_ids, matched_modules


def loss_for_output(output, y, task_type, loss_fn, device):
    if task_type == "classification":
        return loss_fn(output, y.long().to(device))
    return loss_fn(output.reshape(-1), y.float().to(device).reshape(-1))


def make_loader(encoded, labels, indices, batch_size, token_ids, shuffle):
    dataset = SequenceDataset(subset_encoded(encoded, indices), labels[indices])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, int(token_ids["pad"])),
    )


def prediction_scores_for_stats(predictions, task_type):
    predictions = np.asarray(predictions)
    if task_type == "classification":
        if predictions.ndim == 2 and predictions.shape[1] >= 2:
            return predictions[:, 1]
        return predictions.reshape(-1)
    return predictions.reshape(-1)


def summarize_split_predictions(split_name, y_true, y_pred, task_type):
    scores = prediction_scores_for_stats(y_pred, task_type)
    metrics = evaluate_predictions(y_true, y_pred, task_type, precision_k=100)
    spearman = metrics["spearman"]
    print(
        "%s diagnostics: target_mean=%.6g target_std=%.6g prediction_mean=%.6g prediction_std=%.6g spearman=%s" % (
            split_name,
            float(np.mean(y_true)) if len(y_true) else float("nan"),
            float(np.std(y_true)) if len(y_true) else float("nan"),
            float(np.mean(scores)) if len(scores) else float("nan"),
            float(np.std(scores)) if len(scores) else float("nan"),
            spearman,
        )
    )
    return metrics, scores


@torch.no_grad()
def evaluate_split(model, encoded, labels, indices, task_type, token_ids, eval_batch_size, device):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return {"loss": np.nan, "predictions": np.asarray([]), "metrics": evaluate_predictions([], [], task_type)}
    loader = make_loader(encoded, labels, indices, eval_batch_size, token_ids, shuffle=False)
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    losses = []
    predictions = []
    model.eval()
    for input_ids, attention_mask, y in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids, attention_mask=attention_mask)
        loss = loss_for_output(output, y, task_type, loss_fn, device)
        losses.append(float(loss.detach().cpu()))
        if task_type == "classification":
            predictions.append(torch.softmax(output, dim=1).detach().cpu().numpy())
        else:
            predictions.append(output.reshape(-1).detach().cpu().numpy())
    y_pred = np.concatenate(predictions, axis=0) if predictions else np.asarray([])
    metrics = evaluate_predictions(labels[indices], y_pred, task_type, precision_k=100)
    return {"loss": float(np.mean(losses)) if losses else np.nan, "predictions": y_pred, "metrics": metrics}


def primary_validation_metric(task_type):
    return "spearman" if task_type == "regression" else "roc_auc"


def metric_is_better(value, best_value):
    if np.isnan(value):
        return False
    if best_value is None or np.isnan(best_value):
        return True
    return float(value) > float(best_value)


def train_one_model(model, encoded, labels, train_idx, task_type, token_ids, args, device, num_epochs, scheduler_epochs, log_prefix):
    train_dataset = SequenceDataset(subset_encoded(encoded, train_idx), labels[train_idx])
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, int(token_ids["pad"])),
    )
    if len(loader) == 0:
        raise ValueError("empty training set")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = int(math.ceil(len(loader) / float(args.gradient_accumulation_steps)))
    total_steps = steps_per_epoch * int(scheduler_epochs)
    scheduler, warmup_steps = cosine_warmup_scheduler(optimizer, total_steps, args.warmup_ratio)
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    use_bf16 = supports_bf16(device)
    print("%s optimizer steps per epoch: %d total_steps=%d warmup_steps=%d bf16=%s" % (
        log_prefix, steps_per_epoch, total_steps, warmup_steps, use_bf16
    ))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(num_epochs) + 1):
        model.train()
        losses = []
        for batch_idx, (input_ids, attention_mask, y) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                output = model(input_ids, attention_mask=attention_mask)
                loss = loss_for_output(output, y, task_type, loss_fn, device)
            (loss / args.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach().cpu()))

            if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == len(loader):
                if args.gradient_clip_norm is not None and args.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(args.gradient_clip_norm),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        epoch_loss = float(np.mean(losses)) if losses else np.nan
        yield epoch, epoch_loss, optimizer.param_groups[0]["lr"]


def select_epoch_by_extrapolative_validation(model, encoded, labels, train_idx, val_idx, task_type, token_ids, args, device):
    metric_name = primary_validation_metric(task_type)
    print("STAGE 1: extrapolative validation / epoch selection")
    print("  stage train mutation orders: 1..%d" % (int(args.train_mutations) - 1))
    print("  stage validation mutation order: %d" % int(args.train_mutations))
    print("  N stage train: %d" % len(train_idx))
    print("  N stage validation: %d" % len(val_idx))
    best = {
        "epoch": None,
        "validation_metric": np.nan,
        "validation_loss": np.nan,
        "train_metric": np.nan,
        "state": None,
    }
    epochs_without_improvement = 0
    for epoch, train_loss, lr in train_one_model(
        model,
        encoded,
        labels,
        train_idx,
        task_type,
        token_ids,
        args,
        device,
        num_epochs=args.max_epochs,
        scheduler_epochs=args.max_epochs,
        log_prefix="Stage 1",
    ):
        train_eval = evaluate_split(model, encoded, labels, train_idx, task_type, token_ids, args.eval_batch_size, device)
        val_eval = evaluate_split(model, encoded, labels, val_idx, task_type, token_ids, args.eval_batch_size, device)
        train_metric = train_eval["metrics"][metric_name]
        val_metric = val_eval["metrics"][metric_name]
        print(
            "Stage 1 epoch %d/%d train_loss=%.6f train_%s=%s val_loss=%.6f val_%s=%s lr=%.6g" % (
                epoch,
                args.max_epochs,
                train_loss,
                metric_name,
                train_metric,
                val_eval["loss"],
                metric_name,
                val_metric,
                lr,
            )
        )
        if metric_is_better(val_metric, best["validation_metric"]):
            best.update(
                {
                    "epoch": int(epoch),
                    "validation_metric": float(val_metric),
                    "validation_loss": float(val_eval["loss"]),
                    "train_metric": float(train_metric),
                    "state": trainable_state_dict(model),
                }
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    "Stage 1 early stopping at epoch %d; best_epoch=%s best_validation_%s=%s" % (
                        epoch,
                        best["epoch"],
                        metric_name,
                        best["validation_metric"],
                    )
                )
                break
    if best["epoch"] is None:
        raise RuntimeError("could not select a validation epoch; validation metric was NaN for every epoch")
    print("Stage 1 selected epoch:")
    print("  best_validation_epoch: %d" % best["epoch"])
    print("  best_validation_%s: %s" % (metric_name, best["validation_metric"]))
    print("  training_%s_at_best_epoch: %s" % (metric_name, best["train_metric"]))
    print("  validation_loss_at_best_epoch: %.6f" % best["validation_loss"])
    return best


def train_final_fixed_epochs(model, encoded, labels, train_idx, task_type, token_ids, args, device, selected_epochs):
    print("STAGE 2: final controlled-extrapolation model")
    print("  final training orders: <= %d" % int(args.train_mutations))
    print("  N final train: %d" % len(train_idx))
    print("  training exactly selected epochs: %d" % int(selected_epochs))
    for epoch, train_loss, lr in train_one_model(
        model,
        encoded,
        labels,
        train_idx,
        task_type,
        token_ids,
        args,
        device,
        num_epochs=int(selected_epochs),
        scheduler_epochs=args.max_epochs,
        log_prefix="Stage 2",
    ):
        print("Stage 2 epoch %d/%d train_loss=%.6f lr=%.6g" % (epoch, int(selected_epochs), train_loss, lr))
    return model


@torch.no_grad()
def predict(model, encoded, labels, indices, task_type, token_ids, eval_batch_size, device):
    return evaluate_split(model, encoded, labels, indices, task_type, token_ids, eval_batch_size, device)["predictions"]


def evaluate_one_order(model, encoded, labels, nmuts, test_order, spec, args, token_ids, device):
    test_idx = np.where(nmuts == int(test_order))[0]
    print("Evaluating mutation order %d, N=%d" % (int(test_order), len(test_idx)))
    if len(test_idx) == 0:
        return None, None, None
    y_true = labels[test_idx]
    split_eval = evaluate_split(model, encoded, labels, test_idx, spec.task_type, token_ids, args.eval_batch_size, device)
    y_pred = split_eval["predictions"]
    if spec.task_type == "classification" and len(np.unique(y_true)) < 2:
        print("[WARNING] Mutation order %d contains only one class; ROC AUC is NA" % int(test_order))
    metrics, scores = summarize_split_predictions("test_order_%d" % int(test_order), y_true, y_pred, spec.task_type)
    pred_df = prediction_dataframe(
        spec,
        args,
        split="test_order_%d" % int(test_order),
        mutation_order=int(test_order),
        indices=test_idx,
        targets=y_true,
        prediction_scores=scores,
    )
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
        }, pred_df, metrics
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
    }, pred_df, metrics


def prediction_dataframe(spec, args, split, mutation_order, indices, targets, prediction_scores):
    return pd.DataFrame(
        {
            "dataset": spec.name,
            "model_name": args.model_name,
            "train_mutations": int(args.train_mutations),
            "split": split,
            "mutation_order": mutation_order,
            "row_index": np.asarray(indices, dtype=int),
            "target": np.asarray(targets).reshape(-1),
            "prediction": np.asarray(prediction_scores).reshape(-1),
        },
        columns=PREDICTION_COLUMNS,
    )


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


def predictions_output_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name("%s_predictions%s" % (output_path.stem, output_path.suffix or ".csv"))


def save_predictions_incremental(prediction_frames, output_path):
    prediction_frames = [frame for frame in prediction_frames if frame is not None and len(frame) > 0]
    if not prediction_frames:
        return
    output_path = predictions_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.concat(prediction_frames, ignore_index=True)
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        for column in PREDICTION_COLUMNS:
            if column not in old_df.columns:
                old_df[column] = np.nan
        out_df = pd.concat([old_df[PREDICTION_COLUMNS], new_df[PREDICTION_COLUMNS]], ignore_index=True)
    else:
        out_df = new_df[PREDICTION_COLUMNS]
    out_df = out_df.drop_duplicates(
        subset=["dataset", "model_name", "train_mutations", "split", "mutation_order", "row_index"],
        keep="last",
    )
    out_df.to_csv(output_path, index=False, columns=PREDICTION_COLUMNS)
    print("Saved predictions to %s" % output_path)


def print_final_diagnosis(task_type, stage_best, final_train_metrics, test_metrics, train_pred_std, test_pred_stds, train_mutations):
    metric_name = primary_validation_metric(task_type)
    final_train_metric = final_train_metrics.get(metric_name, np.nan)
    test_values = [metrics.get(metric_name, np.nan) for metrics in test_metrics if metrics is not None]
    finite_tests = [float(value) for value in test_values if not np.isnan(value)]
    mean_test = float(np.mean(finite_tests)) if finite_tests else np.nan
    collapse_stds = [std for std in [train_pred_std] + list(test_pred_stds) if not np.isnan(std)]
    collapsed = bool(collapse_stds) and max(collapse_stds) < 1e-6
    print("Diagnostic summary:")
    print("  Stage 1 best validation %s: %s" % (metric_name, stage_best["validation_metric"]))
    print("  Final training %s: %s" % (metric_name, final_train_metric))
    print("  Mean held-out test %s: %s" % (metric_name, mean_test))
    print("  A. LoRA does not fit the low-order training data: %s" % (bool(not np.isnan(final_train_metric) and final_train_metric < 0.2)))
    print("  B. LoRA fits training and order-%d validation but fails at orders >=%d: %s" % (
        int(train_mutations),
        int(train_mutations) + 1,
        bool(
            not np.isnan(final_train_metric)
            and not np.isnan(stage_best["validation_metric"])
            and final_train_metric >= 0.2
            and stage_best["validation_metric"] >= 0.2
            and (np.isnan(mean_test) or mean_test < 0.2)
        ),
    ))
    print("  C. predictions collapse: %s" % collapsed)
    print("  D. apparent implementation problem from sanity checks: False")
    print("  E. LoRA genuinely performs poorly under controlled extrapolation: %s" % (
        bool(
            not collapsed
            and not np.isnan(stage_best["validation_metric"])
            and stage_best["validation_metric"] >= 0.2
            and (np.isnan(mean_test) or mean_test < 0.2)
        )
    ))


def print_run_config(args, device):
    print("LoRA configuration:")
    print("  lora_scope: %s" % args.lora_scope)
    print("  lora_rank: %d" % args.lora_rank)
    print("  lora_alpha: %d" % args.lora_alpha)
    print("  lora_dropout: %.4g" % args.lora_dropout)
    print("  lora_target_modules: %s" % " ".join(args.lora_target_modules))
    print("Optimization configuration:")
    print("  optimizer: AdamW")
    print("  learning_rate: %.6g" % args.learning_rate)
    print("  weight_decay: %.6g" % args.weight_decay)
    print("  scheduler: cosine")
    print("  warmup_ratio: %.6g" % args.warmup_ratio)
    print("  max_epochs: %d" % args.max_epochs)
    print("  early_stopping_patience: %d" % args.early_stopping_patience)
    print("  gradient_clip_norm: %s" % args.gradient_clip_norm)
    print("  batch_size: %d" % args.batch_size)
    print("  eval_batch_size: %d" % args.eval_batch_size)
    print("  gradient_accumulation_steps: %d" % args.gradient_accumulation_steps)
    print("  seed: %d" % args.seed)
    print("  device: %s" % device)
    print("  bf16_supported: %s" % supports_bf16(device))


def initialize_plain_backbone_for_tokenization(model_name, device):
    resolved_model_name = resolve_model_name(model_name)
    print("Model for tokenization: %s" % resolved_model_name)
    plm_init(str(ROOT_DIR))
    return plmEmbeddingModel(plm_name=resolved_model_name, emb_only=True, device=device).to(device)


def build_extrapolative_validation_indices(nmuts, train_mutations):
    train_mutations = int(train_mutations)
    stage_train_idx = np.where((nmuts > 0) & (nmuts < train_mutations))[0]
    stage_val_idx = np.where(nmuts == train_mutations)[0]
    if len(stage_train_idx) == 0:
        raise ValueError("empty extrapolative validation training set; expected variants with 0 < num_mutations < %d" % train_mutations)
    if len(stage_val_idx) == 0:
        raise ValueError("empty extrapolative validation set; expected variants with num_mutations == %d" % train_mutations)
    return stage_train_idx, stage_val_idx


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_mutations", type=int, required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", "--epochs", dest="max_epochs", type=int, default=20)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_scope", choices=["qv", "all_linear"], default="all_linear")
    parser.add_argument("--lora_target_modules", nargs="+", default=["query", "value"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenized_path", default=None)
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if args.max_epochs <= 0:
        raise ValueError("--max_epochs must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience must be non-negative")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be non-negative")
    if not (0 <= args.warmup_ratio < 1):
        raise ValueError("--warmup_ratio must be in [0, 1)")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if args.lora_rank <= 0:
        raise ValueError("--lora_rank must be positive")
    if args.lora_alpha <= 0:
        raise ValueError("--lora_alpha must be positive")
    if args.lora_dropout < 0:
        raise ValueError("--lora_dropout must be non-negative")
    if args.lora_scope == "qv" and not args.lora_target_modules:
        raise ValueError("--lora_target_modules must contain at least one module name when --lora_scope qv")
    if not args.lora_target_modules:
        raise ValueError("--lora_target_modules must contain at least one module name")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print("Using device: %s" % device)
    print_run_config(args, device)
    spec, df, labels, nmuts = load_prepare_dataset(args.dataset)
    train_idx, test_orders = build_controlled_indices(nmuts, args.train_mutations)
    print("Training orders: <= %d" % int(args.train_mutations))
    print("N train: %d" % len(train_idx))
    print("Held-out orders: %s" % (", ".join(map(str, test_orders)) if test_orders else "<none>"))

    pooling_positions = infer_pooling_positions(df, spec)
    if pooling_positions is not None:
        print("Pooling designed positions: %s" % ", ".join(map(str, pooling_positions)))

    encoded = load_pretokenized(args.tokenized_path, len(df))
    if encoded is None:
        tokenization_backbone = initialize_plain_backbone_for_tokenization(args.model_name, device)
        encoded = tokenize_sequences(df, spec, tokenization_backbone, resolve_model_name(args.model_name))
        del tokenization_backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dim = int(np.max(labels)) + 1 if spec.task_type == "classification" else 1
    stage_train_idx, stage_val_idx = build_extrapolative_validation_indices(nmuts, args.train_mutations)

    set_seed(args.seed)
    stage_model, stage_backbone, stage_token_ids, _stage_matched_modules = initialize_plm_lora_head(
        args.model_name,
        spec.task_type,
        output_dim,
        device,
        args,
        pooling_positions,
        encoded=encoded,
        labels=labels,
        diagnostic_indices=stage_train_idx,
        report_init_equivalence=True,
    )
    stage_best = select_epoch_by_extrapolative_validation(
        stage_model,
        encoded,
        labels,
        stage_train_idx,
        stage_val_idx,
        spec.task_type,
        stage_token_ids,
        args,
        device,
    )
    del stage_model, stage_backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(args.seed)
    model, backbone, token_ids, _matched_modules = initialize_plm_lora_head(
        args.model_name,
        spec.task_type,
        output_dim,
        device,
        args,
        pooling_positions,
        encoded=encoded,
        labels=labels,
        diagnostic_indices=train_idx,
        report_init_equivalence=True,
    )
    model = train_final_fixed_epochs(
        model,
        encoded,
        labels,
        train_idx,
        spec.task_type,
        token_ids,
        args,
        device,
        selected_epochs=stage_best["epoch"],
    )

    prediction_frames = []
    test_metrics = []
    test_pred_stds = []
    final_train_eval = evaluate_split(model, encoded, labels, train_idx, spec.task_type, token_ids, args.eval_batch_size, device)
    final_train_metrics, final_train_scores = summarize_split_predictions(
        "final_train_le_%d" % int(args.train_mutations),
        labels[train_idx],
        final_train_eval["predictions"],
        spec.task_type,
    )
    final_train_pred_std = float(np.std(final_train_scores)) if len(final_train_scores) else np.nan
    prediction_frames.append(
        prediction_dataframe(
            spec,
            args,
            split="final_train_le_%d" % int(args.train_mutations),
            mutation_order=np.nan,
            indices=train_idx,
            targets=labels[train_idx],
            prediction_scores=final_train_scores,
        )
    )
    save_predictions_incremental(prediction_frames, args.output_path)

    for test_order in test_orders:
        row, pred_df, metrics = evaluate_one_order(model, encoded, labels, nmuts, test_order, spec, args, token_ids, device)
        if row is not None:
            save_result_incremental(row, args.output_path)
        if pred_df is not None:
            prediction_frames.append(pred_df)
            test_pred_stds.append(float(np.std(pred_df["prediction"].to_numpy(dtype=float))) if len(pred_df) else np.nan)
        if metrics is not None:
            test_metrics.append(metrics)
        save_predictions_incremental(prediction_frames, args.output_path)
    print_final_diagnosis(
        spec.task_type,
        stage_best,
        final_train_metrics,
        test_metrics,
        final_train_pred_std,
        test_pred_stds,
        args.train_mutations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
