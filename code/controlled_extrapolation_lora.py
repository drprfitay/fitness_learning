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


def initialize_plm_lora_head(model_name, task_type, output_dim, device, args, pooling_positions):
    resolved_model_name = resolve_model_name(model_name)
    print("Model: %s" % resolved_model_name)
    plm_init(str(ROOT_DIR))
    backbone = plmEmbeddingModel(plm_name=resolved_model_name, emb_only=True, device=device).to(device)
    _resolved_targets, matched_modules = configure_lora_parameters(backbone.plm, args)
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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = int(math.ceil(len(loader) / float(args.gradient_accumulation_steps)))
    total_steps = steps_per_epoch * int(args.max_epochs)
    scheduler, warmup_steps = cosine_warmup_scheduler(optimizer, total_steps, args.warmup_ratio)
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    use_bf16 = supports_bf16(device)
    print("Fitting LoRA...")
    print("Optimizer steps per epoch: %d total_steps=%d warmup_steps=%d bf16=%s" % (
        steps_per_epoch, total_steps, warmup_steps, use_bf16
    ))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    global_step = 0
    for epoch in range(1, args.max_epochs + 1):
        losses = []
        for batch_idx, (input_ids, attention_mask, y) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                output = model(input_ids, attention_mask=attention_mask)
                if task_type == "classification":
                    loss = loss_fn(output, y.long().to(device))
                else:
                    loss = loss_fn(output.reshape(-1), y.float().to(device).reshape(-1))
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
                global_step += 1

        epoch_loss = float(np.mean(losses))
        print("Epoch %d/%d loss=%.6f lr=%.6g" % (
            epoch, args.max_epochs, epoch_loss, optimizer.param_groups[0]["lr"]
        ))
        if epoch_loss < best_loss - 1e-8:
            best_loss = epoch_loss
            best_state = trainable_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print("Early stopping at epoch %d; best_loss=%.6f" % (epoch, best_loss))
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        print("Restored best LoRA/head state from training epoch loss %.6f" % best_loss)
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

    output_dim = int(np.max(labels)) + 1 if spec.task_type == "classification" else 1
    model, backbone = initialize_plm_lora_head(
        args.model_name,
        spec.task_type,
        output_dim,
        device,
        args,
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
