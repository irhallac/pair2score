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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import PipelineConfig, load_pipeline_config
from .data import _first_existing
from .models import attach_lora, load_backbone, load_tokenizer
from .utils import (
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
    tokens_a = tokenizer(
        texts_a,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens_b = tokenizer(
        texts_b,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.stack([item["label"] for item in batch])
    meta = {
        "id_a": [item["id_a"] for item in batch],
        "id_b": [item["id_b"] for item in batch],
        "len_a": [item["len_a"] for item in batch],
        "len_b": [item["len_b"] for item in batch],
        "longer": [item["longer"] for item in batch],
    }
    return {
        "input_ids_a": tokens_a["input_ids"],
        "attention_mask_a": tokens_a["attention_mask"],
        "input_ids_b": tokens_b["input_ids"],
        "attention_mask_b": tokens_b["attention_mask"],
        "labels": labels,
        "meta": meta,
    }


class ProgressPairwiseEssayDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        essays: pd.DataFrame,
        text_col: str,
        label_col: str,
        id_col: str,
    ):
        self.pairs = pairs.reset_index(drop=True)
        self.essays = essays.set_index(id_col)
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        row = self.pairs.iloc[idx]
        a = self.essays.loc[row["id_a"]]
        b = self.essays.loc[row["id_b"]]
        return {
            "text_a": a[self.text_col],
            "text_b": b[self.text_col],
            "score_a": float(a[self.label_col]),
            "score_b": float(b[self.label_col]),
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "id_a": row["id_a"],
            "id_b": row["id_b"],
            "len_a": int(row["len_a"]),
            "len_b": int(row["len_b"]),
            "longer": row["longer"],
        }


@dataclass
class RelativeArtifacts:
    adapter_dir: Path
    head_path: Path
    embedding_path: Optional[Path]


def train_relative(config: PipelineConfig, config_path: Path) -> RelativeArtifacts:
    set_global_seed(config.seed, cuda_device=config.cuda_device)
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

    length_lookup = {
        str(row[id_col]): len(str(row[text_col]).split())
        for _, row in df[[id_col, text_col]].iterrows()
    }

    def load_pairs(split: str, limit: Optional[int]) -> pd.DataFrame:
        records = []
        taken = 0
        with pairs_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                if entry.get("split") != split:
                    continue
                len_a = entry.get("len_a")
                len_b = entry.get("len_b")
                if len_a is None:
                    len_a = length_lookup.get(str(entry["a"]), 0)
                if len_b is None:
                    len_b = length_lookup.get(str(entry["b"]), 0)
                longer = entry.get("longer")
                if not longer:
                    if len_a > len_b:
                        longer = "a"
                    elif len_b > len_a:
                        longer = "b"
                    else:
                        longer = "equal"
                records.append(
                    {
                        "id_a": str(entry["a"]),
                        "id_b": str(entry["b"]),
                        "label": float(entry["y"]),
                        "len_a": int(len_a),
                        "len_b": int(len_b),
                        "longer": longer,
                    }
                )
                taken += 1
                if limit is not None and taken >= limit:
                    break
        return pd.DataFrame(records)

    max_pairs = config.get("stage1_relative", "pair_sampling", "max_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    val_max_pairs = config.get("stage1_relative", "pair_sampling", "val_max_pairs")
    val_max_pairs = int(val_max_pairs) if val_max_pairs is not None else None
    test_max_pairs = config.get("stage1_relative", "pair_sampling", "test_max_pairs")
    test_max_pairs = int(test_max_pairs) if test_max_pairs is not None else None

    train_pairs_df = load_pairs("train", max_pairs)
    val_pairs_df = load_pairs("val", val_max_pairs)
    test_pairs_df = load_pairs("test", test_max_pairs)

    if train_pairs_df.empty:
        raise ValueError(f"No training pairs found in {pairs_file} for split 'train'.")

    print(
        "[relative] loaded pairs -> "
        f"train:{len(train_pairs_df)} val:{len(val_pairs_df)} test:{len(test_pairs_df)}",
        flush=True,
    )

    tokenizer = load_tokenizer(config.get("model", "base_model"))
    backbone = load_backbone(config.get("model", "base_model")).to(device)
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

    train_dataset = ProgressPairwiseEssayDataset(train_pairs_df, train_df, text_col, label_col, id_col)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_pairs(batch, tokenizer, max_length=512),
    )
    val_loader = None
    if not val_pairs_df.empty:
        val_dataset = ProgressPairwiseEssayDataset(val_pairs_df, val_df, text_col, label_col, id_col)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_pairs(batch, tokenizer, max_length=512),
        )
    test_loader = None
    if not test_pairs_df.empty:
        test_dataset = ProgressPairwiseEssayDataset(test_pairs_df, test_df, text_col, label_col, id_col)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_pairs(batch, tokenizer, max_length=512),
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
        for step, batch in enumerate(iterator, 1):
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

            if train and not disable_progress and total_items:
                interim_loss = total_loss / total_items
                interim_acc = correct / total_items
                print(
                    f"[relative-progress] step {step}/{len(loader)} "
                    f"train_loss={interim_loss:.4f} "
                    f"train_acc={interim_acc:.4f}",
                    flush=True,
                )

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

    predictions = []
    predictions_path = None

    def append_predictions(batch, probs, preds, labels_cpu):
        meta = batch.get("meta", {})
        ids_a = meta.get("id_a", [])
        ids_b = meta.get("id_b", [])
        lens_a = meta.get("len_a", [])
        lens_b = meta.get("len_b", [])
        longer = meta.get("longer", [])
        for i in range(len(ids_a)):
            predictions.append(
                {
                    "id_a": ids_a[i],
                    "id_b": ids_b[i],
                    "len_a": lens_a[i],
                    "len_b": lens_b[i],
                    "longer": longer[i],
                    "label": float(labels_cpu[i].item()),
                    "prob": float(probs[i].item()),
                    "prediction": float(preds[i].item()),
                }
            )

    def run_test(loader):
        if loader is None:
            return {"loss": float("nan"), "acc": float("nan"), "pairs": 0}
        total_loss = 0.0
        total_items = 0
        correct = 0
        model.eval()
        head.eval()
        with torch.no_grad():
            for batch in loader:
                tokens_a = batch["input_ids_a"].to(device)
                attn_a = batch["attention_mask_a"].to(device)
                tokens_b = batch["input_ids_b"].to(device)
                attn_b = batch["attention_mask_b"].to(device)
                labels = batch["labels"].to(device)
                out_a = model(input_ids=tokens_a, attention_mask=attn_a, output_hidden_states=True)
                out_b = model(input_ids=tokens_b, attention_mask=attn_b, output_hidden_states=True)
                emb_a = mean_pool(out_a.hidden_states[-1], attn_a)
                emb_b = mean_pool(out_b.hidden_states[-1], attn_b)
                logits = head(emb_a, emb_b)
                probs = torch.sigmoid(logits).detach().cpu()
                preds = (probs >= 0.5).float()
                loss = loss_fn(logits, labels)

                batch_size_local = labels.size(0)
                total_loss += loss.detach().item() * batch_size_local
                total_items += batch_size_local
                correct += (preds == labels.detach().cpu()).sum().item()
                labels_cpu = labels.detach().cpu()
                append_predictions(batch, probs, preds, labels_cpu)

        if total_items == 0:
            return {"loss": float("nan"), "acc": float("nan"), "pairs": 0}
        return {
            "loss": total_loss / total_items,
            "acc": correct / total_items,
            "pairs": total_items,
        }

    test_metrics = run_test(test_loader)
    if test_metrics["pairs"] > 0:
        print(
            f"[relative] test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"pairs={int(test_metrics['pairs'])}"
        )
        if predictions:
            output_dir = Path(config.get("paths", "output_dir")) / f"run{config.run_id}" / config.trait
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = format_run_stamp().replace(" ", "_").replace(":", "")
            predictions_path = output_dir / f"relative_test_predictions_{timestamp}.csv"
            pd.DataFrame(predictions).to_csv(predictions_path, index=False)
            print(f"[relative] saved detailed predictions to {predictions_path}")

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
    if predictions_path is not None:
        metadata["predictions_file"] = str(predictions_path)
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
