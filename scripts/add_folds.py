#!/usr/bin/env python3
"""Attach Pair2Score fold labels to Kaggle train.csv.

Example:
    python scripts/add_folds.py \
        --input /path/to/train.csv \
        --fold-map data/folds/fold_map.json \
        --output data/datasets/main/train_with_folds.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append fold column using provided mapping")
    parser.add_argument("--input", "-i", default="data/datasets/main/train.csv",
                        help="Path to Kaggle train.csv (default: data/datasets/main/train.csv)")
    parser.add_argument("--fold-map", "-m", default="data/folds/fold_map.json",
                        help="JSON mapping of text_id -> fold (default: data/folds/fold_map.json)")
    parser.add_argument("--output", "-o", default="data/datasets/main/train_with_folds.csv",
                        help="Destination CSV path (default: data/datasets/main/train_with_folds.csv)")
    parser.add_argument("--id-col", default="text_id", help="Identifier column name (default: text_id)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_path = Path(args.fold_map)
    if not fold_path.exists():
        raise SystemExit(f"Fold map not found: {fold_path}")

    with fold_path.open("r", encoding="utf-8") as handle:
        fold_map = json.load(handle)

    df = pd.read_csv(args.input)
    if args.id_col not in df.columns:
        raise SystemExit(f"Column '{args.id_col}' missing from {args.input}")

    df["fold"] = df[args.id_col].map(fold_map)
    missing = df[df["fold"].isna()][args.id_col].tolist()
    if missing:
        raise SystemExit(f"Fold labels missing for {len(missing)} essays (sample: {missing[:5]})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} with {len(df)} rows")


if __name__ == "__main__":
    main()
