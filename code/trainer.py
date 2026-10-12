#!/usr/bin/env python3
"""Simple MLM fine-tuning over aligned protein sequences."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from plm_base import load_model, plm_init


IGNORE_INDEX = -100


def read_aligned_sequences(msa_path: str | Path) -> list[str]:
    """Read aligned sequences from FASTA or one-sequence-per-line text."""
    msa_path = Path(msa_path)
    sequences: list[str] = []
    current: list[str] = []
    saw_fasta = False

    with msa_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                saw_fasta = True
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            if saw_fasta:
                current.append(line)
            else:
                sequences.append(line)

    if current:
        sequences.append("".join(current))

    sequences = [normalize_aligned_sequence(seq) for seq in sequences]
    if not sequences:
        raise ValueError(f"no sequences found in {msa_path}")

    lengths = {len(seq) for seq in sequences}
    if len(lengths) != 1:
        raise ValueError(f"aligned sequences must all have the same length, got {sorted(lengths)}")

    return sequences


def normalize_aligned_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("_", "-")


class MsaMlmDataset(Dataset):
    def __init__(self, sequences: Iterable[str], encode):
        self.encoded = [torch.tensor(encode(seq), dtype=torch.long) for seq in sequences]
        if not self.encoded:
            raise ValueError("cannot train on an empty sequence set")

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.encoded[idx]


def _tokenizer_id(tokenizer, names: tuple[str, ...], attrs: tuple[str, ...]) -> int | None:
    for attr in attrs:
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)

    for name in names:
        if hasattr(tokenizer, "get_idx"):
            try:
                value = tokenizer.get_idx(name)
                if value is not None:
                    return int(value)
            except Exception:
                pass

        if hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                value = tokenizer.convert_tokens_to_ids(name)
                unk = getattr(tokenizer, "unk_token_id", None)
                if value is not None and value != unk:
                    return int(value)
            except Exception:
                pass

        if hasattr(tokenizer, "token_to_id"):
            try:
                value = tokenizer.token_to_id(name)
                if value is not None:
                    return int(value)
            except Exception:
                pass

        if hasattr(tokenizer, "get_vocab"):
            try:
                vocab = tokenizer.get_vocab()
                if name in vocab:
                    return int(vocab[name])
            except Exception:
                pass

    return None


def resolve_token_ids(tokenizer) -> dict[str, int | None]:
    return {
        "pad": _tokenizer_id(tokenizer, ("<pad>", "[PAD]"), ("padding_idx", "pad_token_id")),
        "mask": _tokenizer_id(tokenizer, ("<mask>", "[MASK]", "<mask_1>"), ("mask_idx", "mask_token_id")),
        "cls": _tokenizer_id(tokenizer, ("<cls>", "[CLS]", "<s>"), ("cls_idx", "cls_token_id")),
        "bos": _tokenizer_id(tokenizer, ("<|bos|>", "<s>"), ("bos_token_id",)),
        "eos": _tokenizer_id(tokenizer, ("<eos>", "<|eos|>", "[SEP]", "</s>"), ("eos_idx", "eos_token_id", "sep_token_id")),
        "gap": _tokenizer_id(tokenizer, ("-", "_"), ()),
    }


def collate_encoded(batch: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(seq.numel() for seq in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        input_ids[i, : seq.numel()] = seq
        attention_mask[i, : seq.numel()] = 1
    return input_ids, attention_mask


def mask_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_ids: dict[str, int | None],
    mask_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_id = token_ids["mask"]
    if mask_id is None:
        raise ValueError("selected PLM tokenizer does not expose a mask token")

    candidates = attention_mask.bool()
    for key in ("pad", "cls", "bos", "eos", "gap"):
        token_id = token_ids.get(key)
        if token_id is not None:
            candidates &= input_ids.ne(int(token_id))

    random_mask = torch.rand(input_ids.shape, device=input_ids.device) < mask_probability
    masked_positions = random_mask & candidates

    for row_idx in range(masked_positions.shape[0]):
        if not masked_positions[row_idx].any() and candidates[row_idx].any():
            valid = torch.where(candidates[row_idx])[0]
            masked_positions[row_idx, valid[torch.randint(valid.numel(), (1,), device=input_ids.device)]] = True

    labels = input_ids.clone()
    labels[~masked_positions] = IGNORE_INDEX

    masked_input_ids = input_ids.clone()
    masked_input_ids[masked_positions] = int(mask_id)
    return masked_input_ids, labels


def forward_logits(forward_func, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    try:
        logits, _hidden = forward_func(input_ids, attention_mask=attention_mask)
    except TypeError:
        logits, _hidden = forward_func(input_ids)

    if logits is None:
        raise ValueError(
            "selected PLM wrapper returned logits=None; MLM fine-tuning requires a model with an LM head"
        )
    return logits


def replace_linear_with_lora(module: torch.nn.Module, r: int, alpha: int, dropout: float) -> None:
    import loralib as lora

    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            replacement = lora.Linear(
                child.in_features,
                child.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias=child.bias is not None,
            ).to(child.weight.device)
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            replace_linear_with_lora(child, r=r, alpha=alpha, dropout=dropout)


def configure_trainable_parameters(
    model: torch.nn.Module,
    mode: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    if mode != "lora":
        raise ValueError("mode must be 'full' or 'lora'")

    import loralib as lora

    replace_linear_with_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    lora.mark_only_lora_as_trainable(model)


def trainable_state_dict(model: torch.nn.Module, mode: str) -> dict[str, torch.Tensor]:
    if mode == "full":
        return {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if "lora_" in name}


def output_files(output_path: str | Path, mode: str) -> tuple[Path, Path, Path]:
    output_path = Path(output_path)
    if output_path.suffix == ".pt":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path = output_path
        config_path = output_path.with_suffix(".config.json")
        losses_path = output_path.with_suffix(".losses.pt")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        weights_name = "final_model.pt" if mode == "full" else "lora_weights.pt"
        weights_path = output_path / weights_name
        config_path = output_path / "training_config.json"
        losses_path = output_path / "losses.pt"
    return weights_path, config_path, losses_path


def fine_tune_mlm(
    plm_name: str,
    msa_path: str | Path,
    output_path: str | Path,
    mode: str = "full",
    plm_base_path: str | Path | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    mask_probability: float = 0.15,
    max_sequences: int | None = None,
    seed: int = 0,
    device: str = "auto",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    num_workers: int = 0,
    print_every: int = 10,
) -> dict[str, object]:
    if mode == "non_lora":
        mode = "full"
    if mode not in {"full", "lora"}:
        raise ValueError("mode must be one of: full, non_lora, lora")

    random.seed(seed)
    torch.manual_seed(seed)

    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    if plm_base_path is None:
        plm_base_path = Path(__file__).resolve().parents[1]

    print(f"[new_trainer] plm_base_path={plm_base_path}")
    print(f"[new_trainer] loading PLM={plm_name}")
    plm_init(str(plm_base_path))
    plm_obj = load_model(plm_name)
    model = plm_obj.get_model().to(resolved_device)
    tokenizer = plm_obj.get_tokenizer()
    encode = plm_obj.get_encode()
    forward_func = plm_obj.get_forward()
    token_ids = resolve_token_ids(tokenizer)

    if token_ids["mask"] is None:
        raise ValueError(f"{plm_name} tokenizer has no mask token; cannot run MLM fine-tuning")
    if token_ids["pad"] is None:
        token_ids["pad"] = 0

    sequences = read_aligned_sequences(msa_path)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    print(f"[new_trainer] loaded aligned sequences={len(sequences)} length={len(sequences[0])}")
    print(f"[new_trainer] token ids={token_ids}")
    print(f"[new_trainer] mode={mode} device={resolved_device} epochs={epochs} batch_size={batch_size}")

    configure_trainable_parameters(model, mode, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[new_trainer] trainable_params={trainable_params} total_params={total_params}")

    dataset = MsaMlmDataset(sequences, encode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_encoded(batch, int(token_ids["pad"])),
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    model.train()
    losses: list[float] = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        for batch_idx, (input_ids, attention_mask) in enumerate(loader, start=1):
            input_ids = input_ids.to(resolved_device)
            attention_mask = attention_mask.to(resolved_device)
            masked_input_ids, labels = mask_batch(input_ids, attention_mask, token_ids, mask_probability)

            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(forward_func, masked_input_ids, attention_mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            epoch_losses.append(loss_value)
            global_step += 1

            if print_every and (batch_idx == 1 or batch_idx % print_every == 0):
                print(
                    f"[new_trainer] epoch={epoch}/{epochs} batch={batch_idx}/{len(loader)} "
                    f"step={global_step} loss={loss_value:.6f}"
                )

        print(f"[new_trainer] epoch={epoch}/{epochs} mean_loss={sum(epoch_losses) / len(epoch_losses):.6f}")

    weights_path, config_path, losses_path = output_files(output_path, mode)
    torch.save(trainable_state_dict(model, mode), weights_path)
    torch.save(torch.tensor(losses), losses_path)

    config = {
        "plm_name": plm_name,
        "msa_path": str(msa_path),
        "output_path": str(output_path),
        "weights_path": str(weights_path),
        "mode": mode,
        "plm_base_path": str(plm_base_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "mask_probability": mask_probability,
        "max_sequences": max_sequences,
        "seed": seed,
        "device": str(resolved_device),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "n_sequences": len(sequences),
        "aligned_length": len(sequences[0]),
        "n_steps": global_step,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    print(f"[new_trainer] saved weights={weights_path}")
    print(f"[new_trainer] saved config={config_path}")
    print(f"[new_trainer] saved losses={losses_path}")

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple aligned-MSA MLM fine-tuning.")
    parser.add_argument("--plm_name", required=True)
    parser.add_argument("--mode", choices=["full", "non_lora", "lora"], default="full")
    parser.add_argument("--msa_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--plm_base_path", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--mask_probability", type=float, default=0.15)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fine_tune_mlm(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
