#!/usr/bin/env python
"""
Summarise pair dataset metadata to confirm runs follow the experimental plan.

Usage:
    python scripts/verify_pair_stats.py
    python scripts/verify_pair_stats.py --trait grammar --sizes pairs pairs_small
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

RUN_ROTATIONS = {
    1: ("A B C D".split(), "E"),
    2: ("B C D E".split(), "A"),
    3: ("C D E A".split(), "B"),
    4: ("D E A B".split(), "C"),
    5: ("E A B C".split(), "D"),
}


def load_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarise_meta(meta: dict) -> dict:
    counts = meta.get("counts_actual", {})
    usage = meta.get("usage_stats", {})
    train_usage = usage.get("train", {})
    gap = meta.get("min_score_gap", meta.get("min_gap"))
    target_usage = meta.get("target_usage")
    return {
        "counts": counts,
        "usage": usage,
        "train_mean_usage": train_usage.get("mean_usage"),
        "train_cap": train_usage.get("per_essay_limit"),
        "target_usage": target_usage,
        "min_gap": gap,
        "pair_folds": meta.get("pair_folds"),
        "heldout": meta.get("heldout_fold"),
        "seed": meta.get("seed"),
    }


def check_rotation(run_id: int, folds: Iterable[str] | None, heldout: str | None) -> bool:
    expected_folds, expected_heldout = RUN_ROTATIONS.get(run_id, (None, None))
    if expected_folds is None:
        return True
    if folds is None or heldout is None:
        return True
    return list(folds) == expected_folds and heldout == expected_heldout


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify pair dataset metadata.")
    parser.add_argument(
        "--root",
        default="data",
        help="Root directory containing pairs/ and pairs_small/",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["pairs", "pairs_small"],
        help="Which subdirectories to check (default: pairs, pairs_small).",
    )
    parser.add_argument(
        "--trait",
        help="Filter by trait (e.g., grammar). When omitted, all traits are reported.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    problems = defaultdict(list)

    for size in args.sizes:
        dir_path = root / size
        if not dir_path.exists():
            print(f"[warn] Skipping missing directory: {dir_path}")
            continue
        print(f"=== {size} ===")
        for meta_path in sorted(dir_path.glob("run*_*.jsonl")):
            # skip jsonl; we only want metadata
            pass
        for meta_path in sorted(dir_path.glob("run*_meta.json")):
            parts = meta_path.stem.split("_")
            if len(parts) < 3:
                print(f"[warn] unexpected filename format: {meta_path.name}")
                continue
            trait = parts[1]
            if args.trait and trait != args.trait:
                continue
            run_id = int(parts[0][3:])
            meta = load_meta(meta_path)
            summary = summarise_meta(meta)
            counts = summary["counts"]
            usage = summary["usage"]
            folds = summary["pair_folds"]
            heldout = summary["heldout"]

            mean_usage = summary["train_mean_usage"]
            if isinstance(mean_usage, (int, float)):
                mean_str = f"{mean_usage:.2f}"
            else:
                mean_str = str(mean_usage)
            print(
                f"{meta_path.stem}: "
                f"train={counts.get('train')} val={counts.get('val')} test={counts.get('test')} | "
                f"target_usage={summary['target_usage']} cap={summary['train_cap']} "
                f"mean={mean_str} | "
                f"seed={summary['seed']} min_gap={summary['min_gap']}"
            )
            if folds or heldout:
                print(f"    folds={folds} heldout={heldout}")

            if not check_rotation(run_id, folds, heldout):
                problems["rotation"].append(meta_path)

    if problems:
        print("\n[issues detected]")
        for tag, paths in problems.items():
            print(f"- {tag}:")
            for path in paths:
                print(f"  * {path}")
    else:
        print("\nAll checked metadata files follow the expected template.")


if __name__ == "__main__":
    main()
