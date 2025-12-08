from __future__ import annotations

from pathlib import Path

from .absolute import train_absolute
from .config import PipelineConfig, load_pipeline_config
from .relative import train_relative


def run_pipeline(config_path: str | Path):
    cfg = load_pipeline_config(config_path)
    artifacts = None
    if cfg.get("stage1_relative", "enabled", default=True):
        artifacts = train_relative(cfg, Path(config_path))
    train_absolute(cfg, artifacts, Path(config_path))
