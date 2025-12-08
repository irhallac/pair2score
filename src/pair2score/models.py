from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    model: torch.nn.Module


def _tok_path(base: str) -> str:
    return base if (Path(base) / "tokenizer_config.json").exists() else str(Path(base) / "tokenizer")


def _mdl_path(base: str) -> str:
    return base if (Path(base) / "config.json").exists() else str(Path(base) / "model")


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tok_path = _tok_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    return tokenizer


def load_backbone(model_name: str) -> torch.nn.Module:
    model_path = _mdl_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def attach_lora(
    model: torch.nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Iterable[str],
) -> torch.nn.Module:
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def load_adapter(model: torch.nn.Module, adapter_dir: Path) -> torch.nn.Module:
    return PeftModel.from_pretrained(model, adapter_dir)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model directory")
    parser.add_argument("--config", help="Pipeline config to reuse base_model path")
    args = parser.parse_args()

    model_name = args.model
    if not model_name and args.config:
        from .config import load_pipeline_config

        cfg = load_pipeline_config(args.config)
        model_name = cfg.get("model", "base_model")

    if not model_name:
        parser.error("Provide --model or --config")

    tokenizer = load_tokenizer(model_name)
    print(f"tokenizer ok ({tokenizer.vocab_size})")

    model = load_backbone(model_name)
    device = next(model.parameters()).device
    print(f"backbone ok ({sum(p.numel() for p in model.parameters()):,} params on {device})")


if __name__ == "__main__":
    _main()
