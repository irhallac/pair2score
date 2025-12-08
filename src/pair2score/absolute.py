from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import PipelineConfig, load_pipeline_config
from .data import AbsoluteEssayDataset, load_splits
from .models import attach_lora, load_adapter, load_backbone, load_tokenizer
from .relative import mean_pool, train_relative
from .utils import configure_determinism, current_hostname, format_run_stamp, set_global_seed, write_metadata_file
from .metrics import compute_regression_metrics
import numpy as np


def encode_dataframe(model, tokenizer, df, text_col: str, device: str, batch_size: int = 8, max_length: int = 512):
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            tokens = tokenizer(
                batch[text_col].tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**tokens, output_hidden_states=True)
            pooled = mean_pool(out.hidden_states[-1], tokens["attention_mask"])
            vectors.append(pooled.cpu())
    return torch.cat(vectors, dim=0)


class AbsoluteHead(nn.Module):
    def __init__(self, hidden_size: int, fusion_dim: Optional[int] = None):
        super().__init__()
        input_dim = hidden_size + (fusion_dim or 0)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2 if input_dim > 256 else input_dim),
            nn.ReLU(),
            nn.Linear(input_dim // 2 if input_dim > 256 else input_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.regressor(features).squeeze(-1)


def collate_absolute(batch, tokenizer, max_length=512):
    texts = [item["text"] for item in batch]
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = torch.stack([item["label"] for item in batch])
    embeddings = torch.stack([item["embedding"] for item in batch]) if "embedding" in batch[0] else None
    return {"tokens": tokenized, "labels": labels, "embeddings": embeddings}


def train_absolute(config: PipelineConfig, artifacts, config_path: Path):
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
    splits = load_splits(
        csv_path,
        config.trait,
        config.train_folds,
        config.test_fold,
        config.val_ratio,
        seed=config.seed,
    )

    tokenizer = load_tokenizer(config.get("model", "base_model"))
    backbone = load_backbone(config.get("model", "base_model")).to(device)
    stage_mode = config.get("stage2_absolute", "mode", default="warm_start")

    checkpoint_dir = Path(config.get("paths", "checkpoint_dir")) / f"run{config.run_id}" / config.trait
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reuse_relative = config.get("stage2_absolute", "lora", "reuse_relative_adapter", default=True)
    print(
        "[absolute] data summary -> "
        f"trait={config.trait} train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} "
        f"seed={config.seed} device={device}",
        flush=True,
    )

    run_metadata = {
        "timestamp": format_run_stamp(),
        "hostname": current_hostname(),
        "stage": "absolute",
        "run_id": config.run_id,
        "trait": config.trait,
        "train_size": len(splits.train),
        "val_size": len(splits.val),
        "test_size": len(splits.test),
        "mode": stage_mode,
        "seed": config.seed,
        "device": device,
        "reuse_relative_adapter": bool(reuse_relative),
        "config_path": str(config_path),
        "deterministic_enabled": deterministic_enabled,
        "deterministic_warn_only": deterministic_warn_only,
        "deterministic_allow_tf32": deterministic_allow_tf32,
    }
    if artifacts is not None:
        run_metadata["relative_adapter_dir"] = str(artifacts.adapter_dir)
    if config.metadata:
        for key, value in config.metadata.items():
            run_metadata[f"meta_{key}"] = value

    if reuse_relative and artifacts is not None:
        model = load_adapter(backbone, artifacts.adapter_dir).to(device)
        lora_cfg = config.get("stage2_absolute", "lora") or {}
        stack_new_adapter = bool(lora_cfg.get("stack_new_adapter", False))
        if stack_new_adapter:
            active_adapters = getattr(model, "active_adapters", None)
            if active_adapters is None:
                stage1_adapters = ["default"]
            elif isinstance(active_adapters, (list, tuple)):
                stage1_adapters = list(active_adapters)
            else:
                stage1_adapters = [str(active_adapters)]
            stacked_name = str(lora_cfg.get("stack_adapter_name", "stage2"))
            stage2_cfg = LoraConfig(
                r=int(lora_cfg.get("new_adapter_rank", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.add_adapter(stacked_name, stage2_cfg)
            model.set_adapter(stage1_adapters + [stacked_name])
            frozen_tensors = 0
            for name, param in model.named_parameters():
                if any(adapter_name in name for adapter_name in stage1_adapters):
                    if param.requires_grad:
                        param.requires_grad_(False)
                        frozen_tensors += 1
            for name, param in model.named_parameters():
                if stacked_name in name:
                    param.requires_grad_(True)
            model.to(device)
            print(
                "[absolute] stacked adapter enabled -> "
                f"stage1={stage1_adapters} frozen_tensors={frozen_tensors} stage2='{stacked_name}'",
                flush=True,
            )
            run_metadata["absolute_stage2_stacked_adapter"] = stacked_name
            run_metadata["absolute_stage1_frozen_tensors"] = frozen_tensors
    else:
        lora_cfg = config.get("stage2_absolute", "lora")
        model = attach_lora(
            backbone,
            r=int(lora_cfg.get("new_adapter_rank", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ).to(device)

    hidden_size = model.get_input_embeddings().embedding_dim

    fusion_dim = None
    train_embeddings = val_embeddings = test_embeddings = None
    if stage_mode == "embedding_fusion":
        print(
            "[absolute] encoding essays for embedding fusion "
            f"(train={len(splits.train)} val={len(splits.val)} test={len(splits.test)})...",
            flush=True,
        )
    if stage_mode == "embedding_fusion":
        if artifacts is None:
            raise ValueError("Embedding fusion requires stage-1 artifacts")
        train_embeddings = encode_dataframe(model, tokenizer, splits.train, splits.text_col, device)
        val_embeddings = encode_dataframe(model, tokenizer, splits.val, splits.text_col, device)
        test_embeddings = encode_dataframe(model, tokenizer, splits.test, splits.text_col, device)
        print("[absolute] embedding fusion tensors ready", flush=True)
        fusion_dim = train_embeddings.size(-1)

    head = AbsoluteHead(hidden_size, fusion_dim=fusion_dim).to(device)

    train_dataset = AbsoluteEssayDataset(splits.train, splits.text_col, splits.label_col, embeddings=train_embeddings)
    val_dataset = AbsoluteEssayDataset(splits.val, splits.text_col, splits.label_col, embeddings=val_embeddings)
    test_dataset = AbsoluteEssayDataset(splits.test, splits.text_col, splits.label_col, embeddings=test_embeddings)

    label_values = np.concatenate(
        [
            splits.train[splits.label_col].to_numpy(copy=True),
            splits.val[splits.label_col].to_numpy(copy=True),
            splits.test[splits.label_col].to_numpy(copy=True),
        ]
    )
    label_min = float(label_values.min())
    label_max = float(label_values.max())

    batch_size_value = config.get("stage2_absolute", "training", "batch_size")
    if batch_size_value is None:
        raise ValueError("Missing stage2_absolute.training.batch_size in config.")
    batch_size = int(batch_size_value)
    collate_fn = lambda batch: collate_absolute(batch, tokenizer, max_length=512)
    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(config.seed + 1)
    test_generator = torch.Generator()
    test_generator.manual_seed(config.seed + 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        generator=val_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        generator=test_generator,
    )

    lr_value = config.get("stage2_absolute", "training", "lr")
    if lr_value is None:
        raise ValueError("Missing stage2_absolute.training.lr in config.")
    weight_decay_value = config.get("stage2_absolute", "training", "weight_decay")
    if weight_decay_value is None:
        raise ValueError("Missing stage2_absolute.training.weight_decay in config.")
    optimizer = AdamW(
        [
            {"params": model.parameters(), "lr": float(lr_value)},
            {"params": head.parameters(), "lr": float(lr_value)},
        ],
        weight_decay=float(weight_decay_value),
    )
    loss_fn = nn.L1Loss()

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        head.train() if train else head.eval()
        total_loss = 0.0
        total_items = 0
        preds, labels = [], []
        for batch in loader:
            tokens = {k: v.to(device) for k, v in batch["tokens"].items()}
            y = batch["labels"].to(device)
            embeddings = batch["embeddings"].to(device) if batch["embeddings"] is not None else None

            with torch.set_grad_enabled(train):
                outputs = model(**tokens, output_hidden_states=True)
                pooled = mean_pool(outputs.hidden_states[-1], tokens["attention_mask"])
                if embeddings is not None:
                    pooled = torch.cat([pooled, embeddings.to(device)], dim=-1)
                preds_batch = head(pooled)
                loss = loss_fn(preds_batch, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.detach().item() * batch_size
            total_items += batch_size
            preds.append(preds_batch.detach().cpu())
            labels.append(y.detach().cpu())
        if not preds:
            return {"loss": float("nan"), "mae": float("nan"), "qwk": float("nan")}
        preds_np = torch.cat(preds).numpy()
        labels_np = torch.cat(labels).numpy()
        metrics = compute_regression_metrics(labels_np, preds_np, min_grade=label_min, max_grade=label_max)
        metrics["loss"] = total_loss / max(total_items, 1)
        return metrics

    epochs_value = config.get("stage2_absolute", "training", "epochs")
    if epochs_value is None:
        raise ValueError("Missing stage2_absolute.training.epochs in config.")
    epochs = int(epochs_value)
    monitor_name = str(config.get("stage2_absolute", "training", "monitor", default="val_qwk")).lower()
    monitor_mode = str(config.get("stage2_absolute", "training", "monitor_mode", default="auto")).lower()
    patience_value = config.get("stage2_absolute", "training", "early_stop_patience", 0)
    patience = int(patience_value) if patience_value is not None else 0
    min_delta_value = config.get("stage2_absolute", "training", "early_stop_min_delta", 0.0)
    min_delta = float(min_delta_value if min_delta_value is not None else 0.0)

    def resolve_metric(metrics: dict[str, float], name: str) -> float:
        if name in metrics:
            return float(metrics[name])
        if name.startswith("val_"):
            key = name[4:]
            if key in metrics:
                return float(metrics[key])
        return float(metrics.get(name, float("nan")))

    if monitor_mode == "auto":
        monitor_mode = "min" if any(token in monitor_name for token in ("loss", "mae")) else "max"

    def improved(current: float, best: float | None) -> bool:
        if np.isnan(current):
            return False
        if best is None or np.isnan(best):
            return True
        if monitor_mode == "min":
            return current < best - min_delta
        return current > best + min_delta

    best_val_metrics = None
    best_metric = None
    best_epoch = None
    stale_epochs = 0
    best_model_state = None
    best_head_state = None
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(train_loader, train=True)
        val_metrics = run_epoch(val_loader, train=False)
        if not sys.stderr.isatty():
            print(
                f"[absolute] epoch {epoch}: "
                f"train_mae={train_metrics.get('mae'):.4f} "
                f"train_qwk={train_metrics.get('qwk'):.4f} "
                f"val_mae={val_metrics.get('mae'):.4f} "
                f"val_qwk={val_metrics.get('qwk'):.4f}",
                flush=True,
            )
        current_metric = resolve_metric(val_metrics, monitor_name)
        if improved(current_metric, best_metric):
            best_metric = current_metric
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            stale_epochs = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            torch.save(
                {
                    "model": model.state_dict(),
                    "head": head.state_dict(),
                    "val_metrics": best_val_metrics,
                    "epoch": best_epoch,
                },
                Path(config.get("paths", "checkpoint_dir")) / f"run{config.run_id}" / config.trait / "absolute_best.pt",
            )
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                if not sys.stderr.isatty():
                    print(f"[absolute] early stop at epoch {epoch} (patience {patience})")
                break

    best_val = None
    if best_val_metrics is not None and best_epoch is not None:
        best_val = {"epoch": best_epoch, **best_val_metrics}

    if best_model_state is not None and best_head_state is not None:
        model.load_state_dict(best_model_state)
        head.load_state_dict(best_head_state)

    test_metrics = run_epoch(test_loader, train=False)
    val_qwk_display = float("nan")
    val_mae_display = float("nan")
    if best_val is not None:
        if isinstance(best_val.get("qwk"), (int, float)):
            val_qwk_display = float(best_val.get("qwk"))
        if isinstance(best_val.get("mae"), (int, float)):
            val_mae_display = float(best_val.get("mae"))
    test_qwk_display = float(test_metrics.get("qwk", float("nan")))
    test_mae_display = float(test_metrics.get("mae", float("nan")))
    print(
        "[absolute] finished: "
        f"best_epoch={best_epoch if best_epoch is not None else 'N/A'} "
        f"val_mae={val_mae_display:.4f} "
        f"val_qwk={val_qwk_display:.4f} "
        f"test_mae={test_mae_display:.4f} "
        f"test_qwk={test_qwk_display:.4f}",
        flush=True,
    )

    run_metadata.update(
        {
            "monitor": monitor_name,
            "monitor_mode": monitor_mode,
            "early_stop_patience": patience,
            "early_stop_min_delta": min_delta,
            "best_epoch": best_epoch,
        }
    )
    if best_val is not None:
        run_metadata["best_val_mae"] = best_val.get("mae")
        run_metadata["best_val_qwk"] = best_val.get("qwk")
    if test_metrics:
        run_metadata["test_mae"] = test_metrics.get("mae")
        run_metadata["test_qwk"] = test_metrics.get("qwk")
    write_metadata_file(checkpoint_dir, run_metadata, filename="absolute_run_info.txt")

    output_dir = Path(config.get("paths", "output_dir")) / f"run{config.run_id}" / config.trait
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "val": best_val,
            "test": test_metrics,
            "config_path": str(config_path),
            "mode": stage_mode,
        },
        output_dir / "metrics.pt",
    )
    metrics_metadata = dict(run_metadata)
    metrics_metadata.update(
        {
            "metrics_timestamp": format_run_stamp(),
            "best_val_mae": None if best_val is None else best_val.get("mae"),
            "best_val_qwk": None if best_val is None else best_val.get("qwk"),
            "test_mae": test_metrics.get("mae"),
            "test_qwk": test_metrics.get("qwk"),
        }
    )
    write_metadata_file(output_dir, metrics_metadata, filename="absolute_metrics_info.txt")


def _main():
    parser = argparse.ArgumentParser(description="Smoke-test absolute trainer.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline config path.")
    parser.add_argument("--epochs", type=int, help="Override training epochs for quick checks.")
    parser.add_argument("--relative-off", action="store_true", help="Skip reusing relative adapter, force fresh LoRA.")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    if args.epochs is not None:
        cfg.raw.setdefault("stage2_absolute", {}).setdefault("training", {})["epochs"] = args.epochs
    if args.relative_off:
        cfg.raw.setdefault("stage2_absolute", {}).setdefault("lora", {})["reuse_relative_adapter"] = False

    artifacts = None
    if cfg.get("stage2_absolute", "lora", "reuse_relative_adapter", True):
        relative_cfg = load_pipeline_config(args.config)
        artifacts = train_relative(relative_cfg, Path(args.config))

    train_absolute(cfg, artifacts, Path(args.config))
    print("absolute training completed")


if __name__ == "__main__":
    _main()
