from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class EssaySplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    text_col: str
    label_col: str
    id_col: str


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Could not find any of {candidates}")


def load_splits(csv_path: Path, trait: str, train_folds: list[str], test_fold: str, val_ratio: float, seed: int = 36) -> "EssaySplits":
    df = pd.read_csv(csv_path)
    text_col = _first_existing(df, ["full_text", "text", "essay"])
    label_col = _first_existing(
        df,
        [
            trait,
            trait.lower(),
            f"{trait}_score",
            f"{trait.lower()}_score",
            "score",
            "overall",
            "mean_score",
        ],
    )
    id_col = _first_existing(df, ["text_id", "essay_id", "id"])
    fold_col = _first_existing(df, ["fold", "Fold", "FOLD"])

    pool = df[df[fold_col].isin(train_folds)].copy()
    test = df[df[fold_col] == test_fold].copy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))
    n_val = max(1, int(len(pool) * val_ratio))
    val_idx = perm[:n_val]
    val = pool.iloc[val_idx].copy()
    train = pool.drop(pool.index[val_idx]).copy()

    return EssaySplits(train=train, val=val, test=test, text_col=text_col, label_col=label_col, id_col=id_col)


def build_pair_dataframe(pool: pd.DataFrame, label_col: str, id_col: str, min_gap: float = 1.0) -> pd.DataFrame:
    pairs = []
    scores = pool[label_col].to_numpy()
    ids = pool[id_col].to_numpy()
    n = len(pool)
    for i in range(n):
        for j in range(i + 1, n):
            gap = scores[i] - scores[j]
            if abs(gap) < min_gap:
                continue
            pairs.append(
                {
                    "id_a": ids[i],
                    "id_b": ids[j],
                    "label": 1 if gap > 0 else 0,
                    "score_a": scores[i],
                    "score_b": scores[j],
                    "gap": float(gap),
                }
            )
    return pd.DataFrame(pairs)


class PairwiseEssayDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, essays: pd.DataFrame, text_col: str, label_col: str, id_col: str):
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
        }


class AbsoluteEssayDataset(Dataset):
    def __init__(self, data: pd.DataFrame, text_col: str, label_col: str, embeddings=None):
        self.data = data.reset_index(drop=True)
        self.text_col = text_col
        self.label_col = label_col
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        item = {
            "text": row[self.text_col],
            "label": torch.tensor(row[self.label_col], dtype=torch.float32),
        }
        if self.embeddings is not None:
            embedding = self.embeddings[idx]
            if isinstance(embedding, torch.Tensor):
                item["embedding"] = embedding.detach().clone().to(dtype=torch.float32)
            else:
                item["embedding"] = torch.as_tensor(embedding, dtype=torch.float32)
        return item
