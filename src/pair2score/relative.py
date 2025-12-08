from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import PipelineConfig, load_pipeline_config
from .data import PairwiseEssayDataset, _first_existing
from .models import attach_lora, load_backbone, load_tokenizer
from .utils import (
    configure_determinism,
    current_hostname,
    format_run_stamp,
    set_global_seed,
    write_metadata_file,
)


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-8)
    return summed / denom


class RelativeScorer(nn.Module):
    """Shared linear head (bias-free) enforcing Δ(a,b) = score(a) − score(b) with antisymmetry."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        score_a = self.scorer(emb_a)
        score_b = self.scorer(emb_b)
        return (score_a - score_b).squeeze(-1)


def collate_pairs(batch: Iterable[dict], tokenizer, max_length: int = 512):
    texts_a = [item["text_a"] for item in batch]
    texts_b = [item["text_b"] for item in batch]
    tokens_a = tokenizer(texts_a, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokens_b = tokenizer(texts_b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = torch.stack([item["label"] for item in batch])
    return {
        "input_ids_a": tokens_a["input_ids"],
        "attention_mask_a": tokens_a["attention_mask"],
        "input_ids_b": tokens_b["input_ids"],
        "attention_mask_b": tokens_b["attention_mask"],
        "labels": labels,
    }


@dataclass
class RelativeArtifacts:
    adapter_dir: Path
    head_path: Path
    embedding_path: Optional[Path]


def train_relative(config: PipelineConfig, config_path: Path) -> RelativeArtifacts:
    set_global_seed(config.seed, cuda_device=config.cuda_device)
    determinism_cfg = config.get("determinism") or {}
    deterministic_enabled = bool(determinism_cfg.get("enabled", False))
    deterministic_warn_only = bool(determinism_cfg.get("warn_only", False))
    deterministic_allow_tf32 = bool(determinism_cfg.get("allow_tf32", True))
    configure_determinism(
        enabled=deterministic_enabled,
        warn_only=deterministic_warn_only,
        allow_tf32=deterministic_allow_tf32,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(config.cuda_device)
        device = f"cuda:{config.cuda_device}"
    else:
        device = "cpu"

    csv_path = Path(config.get("paths", "essays_csv"))
    df = pd.read_csv(csv_path)
    text_col = _first_existing(df, ["full_text", "text", "essay"])
    label_col = _first_existing(
        df,
        [
            config.trait,
            config.trait.lower(),
            f"{config.trait}_score",
            f"{config.trait.lower()}_score",
            "score",
            "overall",
            "mean_score",
        ],
    )
    id_col = _first_existing(df, ["text_id", "essay_id", "id"])

    pair_cache_root = config.get("paths", "pair_cache_dir", default="data/pairs_large")
    pairs_dir = Path(pair_cache_root)
    pairs_file = config.get("stage1_relative", "pairs_file")
    if pairs_file is None:
        pairs_file = pairs_dir / f"run{config.run_id}_{config.trait}.jsonl"
    else:
        pairs_file = Path(pairs_file)
    meta_file = config.get("stage1_relative", "pairs_meta")
    if meta_file is None:
        meta_file = pairs_file.with_name(pairs_file.stem + "_meta.json")
    else:
        meta_file = Path(meta_file)

    if not pairs_file.exists():
        raise FileNotFoundError(f"Pair file not found: {pairs_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Pair metadata not found: {meta_file}")

    with meta_file.open("r", encoding="utf-8") as handle:
        pairs_meta = json.load(handle)

    split_ids = pairs_meta.get("splits", {})
    train_ids = set(split_ids.get("train", []))
    val_ids = set(split_ids.get("val", []))
    test_ids = set(split_ids.get("test", []))

    train_df = df[df[id_col].astype(str).isin(train_ids)].copy()
    val_df = df[df[id_col].astype(str).isin(val_ids)].copy()
    test_df = df[df[id_col].astype(str).isin(test_ids)].copy()

    for frame in (train_df, val_df, test_df):
        if not frame.empty:
            frame[id_col] = frame[id_col].astype(str)

    def load_pairs(split: str, limit: Optional[int]) -> pd.DataFrame:
        records = []
        taken = 0
        with pairs_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                if entry.get("split") != split:
                    continue
                records.append(
                    {
                        "id_a": str(entry["a"]),
                        "id_b": str(entry["b"]),
                        "label": float(entry["y"]),
                    }
                )
                taken += 1
                if limit is not None and taken >= limit:
                    break
        return pd.DataFrame(records)

    sampling_cfg = config.get("stage1_relative", "pair_sampling") or {}

    subset_flag = sampling_cfg.get("subset_mode")
    if subset_flag is None:
        subset_flag = sampling_cfg.get("smoke_test")
    if subset_flag is None:
        subset_flag = sampling_cfg.get("enabled")
    if subset_flag is None:
        subset_flag = any(key in sampling_cfg for key in ("max_pairs", "val_max_pairs", "test_max_pairs"))
    use_limits = bool(subset_flag)

    def resolve_limit(name: str):
        value = sampling_cfg.get(name)
        if not use_limits or value in (None, "all"):
            return None
        return int(value)

    max_pairs = resolve_limit("max_pairs")
    val_max_pairs = resolve_limit("val_max_pairs")
    test_max_pairs = resolve_limit("test_max_pairs")

    train_pairs_df = load_pairs("train", max_pairs)
    val_pairs_df = load_pairs("val", val_max_pairs)
    test_pairs_df = load_pairs("test", test_max_pairs)

    if train_pairs_df.empty:
        raise ValueError(f"No training pairs found in {pairs_file} for split 'train'.")

    print(
        "[relative] data summary -> "
        f"trait={config.trait} train_pairs={len(train_pairs_df)} val_pairs={len(val_pairs_df)} test_pairs={len(test_pairs_df)} "
        f"seed={config.seed} device={device}",
        flush=True,
    )

    tokenizer = load_tokenizer(config.get("model", "base_model"))
    backbone = load_backbone(config.get("model", "base_model")).to(device)
    # Attach LoRA adapters so both towers share the same Llama backbone (Siamese).
    lora_cfg = config.get("stage1_relative", "lora")
    model = attach_lora(
        backbone,
        r=int(lora_cfg.get("r", 16)),
        alpha=int(lora_cfg.get("alpha", 32)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ).to(device)

    hidden_size = model.get_input_embeddings().embedding_dim
    head = RelativeScorer(hidden_size).to(device)

    batch_size_value = config.get("stage1_relative", "training", "batch_size")
    if batch_size_value is None:
        raise ValueError("Missing stage1_relative.training.batch_size in config.")
    batch_size = int(batch_size_value)

    train_dataset = PairwiseEssayDataset(train_pairs_df, train_df, text_col, label_col, id_col)
    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(config.seed + 1)
    test_generator = torch.Generator()
    test_generator.manual_seed(config.seed + 2)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_pairs(batch, tokenizer, max_length=512),
        generator=train_generator,
    )

    val_loader = None
    if not val_pairs_df.empty:
        val_dataset = PairwiseEssayDataset(val_pairs_df, val_df, text_col, label_col, id_col)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_pairs(batch, tokenizer, max_length=512),
            generator=val_generator,
        )

    test_loader = None
    if not test_pairs_df.empty:
        test_dataset = PairwiseEssayDataset(test_pairs_df, test_df, text_col, label_col, id_col)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_pairs(batch, tokenizer, max_length=512),
            generator=test_generator,
        )

    lr_value = config.get("stage1_relative", "training", "lr")
    if lr_value is None:
        raise ValueError("Missing stage1_relative.training.lr in config.")
    weight_decay_value = config.get("stage1_relative", "training", "weight_decay")
    if weight_decay_value is None:
        raise ValueError("Missing stage1_relative.training.weight_decay in config.")
    optimizer = AdamW(
        [
            {"params": model.parameters(), "lr": float(lr_value)},
            {"params": head.parameters(), "lr": float(lr_value)},
        ],
        weight_decay=float(weight_decay_value),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    def run_epoch(loader, train: bool, *, show_progress: bool, desc: str):
        if loader is None:
            return {"loss": float("nan"), "acc": float("nan"), "pairs": 0}
        iterator = loader
        if show_progress and not disable_progress:
            iterator = tqdm(loader, desc=desc, leave=False)
        total_loss = 0.0
        total_items = 0
        correct = 0
        model.train() if train else model.eval()
        head.train() if train else head.eval()
        for batch in iterator:
            optimizer.zero_grad() if train else None
            input_ids_a = batch["input_ids_a"].to(device)
            attention_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_b = batch["attention_mask_b"].to(device)
            labels = batch["labels"].to(device)

            with torch.set_grad_enabled(train):
                out_a = model(input_ids=input_ids_a, attention_mask=attention_a, output_hidden_states=True)
                out_b = model(input_ids=input_ids_b, attention_mask=attention_b, output_hidden_states=True)
                emb_a = mean_pool(out_a.hidden_states[-1], attention_a)
                emb_b = mean_pool(out_b.hidden_states[-1], attention_b)
                logits = head(emb_a, emb_b)
                loss = loss_fn(logits, labels)
                if train:
                    loss.backward()
                    optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            batch_size_local = labels.size(0)
            total_loss += loss.detach().item() * batch_size_local
            total_items += batch_size_local
            correct += (preds == labels).sum().item()

        if total_items == 0:
            return {"loss": float("nan"), "acc": float("nan"), "pairs": 0}
        return {
            "loss": total_loss / total_items,
            "acc": correct / total_items,
            "pairs": total_items,
        }

    epochs_value = config.get("stage1_relative", "training", "epochs")
    if epochs_value is None:
        raise ValueError("Missing stage1_relative.training.epochs in config.")
    epochs = int(epochs_value)
    disable_progress = not sys.stderr.isatty()

    monitor_name = str(config.get("stage1_relative", "training", "monitor", default="val_acc")).lower()
    monitor_mode = str(config.get("stage1_relative", "training", "monitor_mode", default="auto")).lower()
    patience_value = config.get("stage1_relative", "training", "early_stop_patience", 0)
    patience = int(patience_value) if patience_value is not None else 0
    min_delta_value = config.get("stage1_relative", "training", "early_stop_min_delta", 0.0)
    min_delta = float(min_delta_value if min_delta_value is not None else 0.0)

    def resolve_metric(metrics: dict[str, float], name: str) -> float:
        if name in metrics:
            return float(metrics[name])
        if name.startswith("val_") and name[4:] in metrics:
            return float(metrics[name[4:]])
        if name.startswith("train_") and name[6:] in metrics:
            return float(metrics[name[6:]])
        return float(metrics.get(name, float("nan")))

    if monitor_mode == "auto":
        monitor_mode = "min" if any(token in monitor_name for token in ("loss",)) else "max"

    def improved(current: float, best: float | None) -> bool:
        if np.isnan(current):
            return False
        if best is None or np.isnan(best):
            return True
        if monitor_mode == "min":
            return current < best - min_delta
        return current > best + min_delta

    best_metric = None
    best_epoch = None
    best_model_state = None
    best_head_state = None
    best_metrics = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(loader, train=True, show_progress=True, desc=f"Relative epoch {epoch}")
        val_metrics = run_epoch(val_loader, train=False, show_progress=False, desc="Relative val")

        combined_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        if val_metrics["pairs"] > 0:
            combined_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        else:
            combined_metrics.update({k: v for k, v in train_metrics.items() if k not in combined_metrics})

        current_metric = resolve_metric(combined_metrics, monitor_name)

        summary = (
            f"[relative] epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f}"
        )
        if val_metrics["pairs"] > 0:
            summary += f" val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        summary += f" monitor={current_metric:.4f}"
        print(summary)

        if improved(current_metric, best_metric):
            best_metric = current_metric
            best_epoch = epoch
            best_metrics = dict(combined_metrics)
            stale_epochs = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                print(f"[relative] early stop at epoch {epoch} (patience {patience})")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        head.load_state_dict(best_head_state)

    test_metrics = run_epoch(test_loader, train=False, show_progress=False, desc="Relative test")
    if test_metrics["pairs"] > 0:
        print(
            f"[relative] test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"pairs={int(test_metrics['pairs'])}"
        )
    best_val_acc = float("nan")
    best_val_loss = float("nan")
    if best_metrics:
        if "val_acc" in best_metrics:
            best_val_acc = float(best_metrics["val_acc"])
        if "val_loss" in best_metrics:
            best_val_loss = float(best_metrics["val_loss"])
    print(
        "[relative] finished: "
        f"best_epoch={best_epoch if best_epoch is not None else 'N/A'} "
        f"val_loss={best_val_loss:.4f} "
        f"val_acc={best_val_acc:.4f} "
        f"test_acc={test_metrics.get('acc', float('nan')):.4f}",
        flush=True,
    )

    output_root = Path(config.get("paths", "checkpoint_dir")) / f"run{config.run_id}" / config.trait / "relative"
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": format_run_stamp(),
        "hostname": current_hostname(),
        "stage": "relative",
        "run_id": config.run_id,
        "trait": config.trait,
        "train_folds": config.train_folds,
        "test_fold": config.test_fold,
        "val_ratio": config.val_ratio,
        "seed": config.seed,
        "device": device,
        "train_pair_count": int(train_pairs_df.shape[0]),
        "val_pair_count": int(val_pairs_df.shape[0]),
        "test_pair_count": int(test_pairs_df.shape[0]),
        "pairs_file": str(pairs_file),
        "pairs_meta": str(meta_file),
        "config_path": str(config_path),
        "deterministic_enabled": deterministic_enabled,
        "deterministic_warn_only": deterministic_warn_only,
        "deterministic_allow_tf32": deterministic_allow_tf32,
    }
    metadata["counts_requested"] = pairs_meta.get("counts_requested", {})
    metadata["counts_actual"] = pairs_meta.get("counts_actual", {})
    metadata.update(
        {
            "monitor": monitor_name,
            "monitor_mode": monitor_mode,
            "early_stop_patience": patience,
            "early_stop_min_delta": min_delta,
            "best_epoch": best_epoch,
        }
    )
    if best_metrics:
        for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
            if key in best_metrics:
                metadata[f"best_{key}"] = best_metrics[key]
    if test_metrics["pairs"] > 0:
        metadata["test_loss"] = test_metrics["loss"]
        metadata["test_acc"] = test_metrics["acc"]
    if config.metadata:
        for key, value in config.metadata.items():
            metadata[f"meta_{key}"] = value
    write_metadata_file(output_root, metadata, filename="relative_run_info.txt")
    adapter_dir = output_root / "adapter"
    model.save_pretrained(adapter_dir)
    head_path = output_root / "relative_head.pt"
    torch.save(head.state_dict(), head_path)

    embedding_path = None
    if config.get("stage1_relative", "outputs", "save_embeddings", default=True):
        model.eval()
        head.eval()
        embedding_path = output_root / "train_embeddings.pt"
        embeddings = []
        ids = []
        train_for_embeddings = train_df
        print(f"[relative] saving train embeddings for {len(train_for_embeddings)} essays...", flush=True)
        with torch.no_grad():
            for _, row in train_for_embeddings.iterrows():
                tokens = tokenizer(
                    row[text_col],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                out = model(**tokens, output_hidden_states=True)
                emb = mean_pool(out.hidden_states[-1], tokens["attention_mask"])
                embeddings.append(emb.cpu())
                ids.append(str(row[id_col]))
        if embeddings:
            stacked = torch.cat(embeddings, dim=0)
            torch.save({"ids": ids, "embeddings": stacked}, embedding_path)
            print(f"[relative] embeddings saved to {embedding_path}", flush=True)

    return RelativeArtifacts(adapter_dir=adapter_dir, head_path=head_path, embedding_path=embedding_path)


def _main():
    parser = argparse.ArgumentParser(description="Smoke-test relative trainer.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline config path.")
    parser.add_argument("--epochs", type=int, help="Override epochs for quick checks.")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    if args.epochs is not None:
        cfg.raw.setdefault("stage1_relative", {}).setdefault("training", {})["epochs"] = args.epochs

    artifacts = train_relative(cfg, Path(args.config))
    print(f"relative artifacts saved under {artifacts.adapter_dir}")


if __name__ == "__main__":
    _main()
