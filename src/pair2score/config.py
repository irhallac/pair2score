from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    raw: Dict[str, Any]

    def get(self, *keys: str, default=None):
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def run_id(self) -> int:
        value = self.get("pipeline", "experiment_id")
        if value is not None:
            return str(value)
        value = self.get("pipeline", "run_id")
        if value is None:
            raise ValueError("Missing required config key: pipeline.experiment_id or pipeline.run_id")
        return str(value)

    @property
    def trait(self) -> str:
        value = self.get("pipeline", "trait")
        if value is None:
            raise ValueError("Missing required config key: pipeline.trait")
        return str(value)

    @property
    def train_folds(self) -> list[str]:
        folds = self.get("pipeline", "train_folds")
        if folds is None:
            raise ValueError("Missing required config key: pipeline.train_folds")
        return list(folds)

    @property
    def test_fold(self) -> str:
        value = self.get("pipeline", "test_fold")
        if value is None:
            raise ValueError("Missing required config key: pipeline.test_fold")
        return str(value)

    @property
    def val_ratio(self) -> float:
        value = self.get("pipeline", "val_ratio")
        if value is None:
            raise ValueError("Missing required config key: pipeline.val_ratio")
        return float(value)

    @property
    def seed(self) -> int:
        value = self.get("pipeline", "seed")
        if value is None:
            raise ValueError("Missing required config key: pipeline.seed")
        return int(value)

    @property
    def metadata(self) -> dict:
        return self.get("metadata") or {}

    @property
    def cuda_device(self) -> int:
        candidates = (
            self.get("hardware", "cuda_device"),
            self.get("pipeline", "cuda_device"),
            self.raw.get("gpu"),
        )
        for value in candidates:
            if value is not None:
                return int(value)
        return 0


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return PipelineConfig(data)
