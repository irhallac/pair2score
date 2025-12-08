#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def split_ids(ids: List[str], val_count: int, test_count: int, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    val_ids = shuffled[:val_count]
    test_ids = shuffled[val_count : val_count + test_count]
    train_ids = shuffled[val_count + test_count :]
    return train_ids, val_ids, test_ids


BIN_NAMES = ("large", "medium", "small")


def build_candidate_bins(scores: np.ndarray, min_gap: float) -> Dict[str, List[Tuple[int, int]]]:
    bins: Dict[str, List[Tuple[int, int]]] = {name: [] for name in BIN_NAMES}
    n = len(scores)
    for i in range(n - 1):
        score_i = scores[i]
        for j in range(i + 1, n):
            gap = abs(score_i - scores[j])
            if gap < min_gap:
                continue
            if gap >= 3.0:
                bins["large"].append((i, j))
            elif gap >= 2.0:
                bins["medium"].append((i, j))
            else:
                bins["small"].append((i, j))
    for pairs in bins.values():
        random.shuffle(pairs)
    return bins


def pair_phase_one(
    bins: Dict[str, List[Tuple[int, int]]],
    usage: Counter,
    selected: Dict[Tuple[int, int], str],
) -> None:
    unused = set(range(len(usage)))

    while unused:
        i = unused.pop()
        found = False
        for bin_name in BIN_NAMES:
            candidates = bins[bin_name]
            for idx, (a, b) in enumerate(candidates):
                if i not in (a, b):
                    continue
                other = b if a == i else a
                if other not in unused:
                    continue
                key = (min(i, other), max(i, other))
                if key in selected:
                    continue
                selected[key] = bin_name
                usage[i] += 1
                usage[other] += 1
                unused.discard(other)
                del candidates[idx]
                found = True
                break
            if found:
                break
        if not found:
            partner_candidates = list(unused) if unused else [idx for idx in range(len(usage)) if idx != i]
            random.shuffle(partner_candidates)
            for other in partner_candidates:
                if other == i:
                    continue
                key = (min(i, other), max(i, other))
                if key in selected:
                    continue
                for bin_name in BIN_NAMES:
                    candidates = bins[bin_name]
                    for idx, (a, b) in enumerate(candidates):
                        if {a, b} == {i, other}:
                            del candidates[idx]
                            break
                selected[key] = "fallback"
                usage[i] += 1
                usage[other] += 1
                unused.discard(other)
                break
            else:
                raise RuntimeError(f"Unable to cover essay index {i}")


def desired_counts(target_total: int, selected: Dict[Tuple[int, int], str], bins: Dict[str, List[Tuple[int, int]]]) -> Dict[str, int]:
    current = Counter(selected.values())
    remaining = target_total - len(selected)
    bin_sizes = {name: len(lst) for name, lst in bins.items()}
    total_candidates = sum(bin_sizes.values())
    if total_candidates == 0:
        return {name: current.get(name, 0) for name in bins}
    ratios = {name: size / total_candidates for name, size in bin_sizes.items()}
    desired = {name: current.get(name, 0) + int(round(remaining * ratios[name])) for name in bins}
    # Adjust to match exactly target_total
    while sum(desired.values()) < target_total:
        name = max(ratios, key=ratios.get)
        desired[name] += 1
    while sum(desired.values()) > target_total:
        name = min(ratios, key=ratios.get)
        if desired[name] > current.get(name, 0):
            desired[name] -= 1
        else:
            break
    return desired


def fill_pairs(
    bins: Dict[str, List[Tuple[int, int]]],
    usage: Counter,
    selected: Dict[Tuple[int, int], str],
    target_total: int,
    max_usage: int,
) -> None:
    desired = desired_counts(target_total, selected, bins)
    bin_order = BIN_NAMES

    while len(selected) < target_total:
        progress = False
        for name in bin_order:
            if len([1 for bin_label in selected.values() if bin_label == name]) >= desired.get(name, 0):
                continue
            candidates = bins[name]
            while candidates:
                i, j = candidates.pop()
                key = (min(i, j), max(i, j))
                if key in selected:
                    continue
                if usage[i] >= max_usage or usage[j] >= max_usage:
                    continue
                selected[key] = name
                usage[i] += 1
                usage[j] += 1
                progress = True
                break
        if not progress:
            # Relax cap and try again
            max_usage += 1
            if max_usage > target_total:
                # no more room to grow; stop early
                break


def materialize_pairs(
    selected: Dict[Tuple[int, int], str],
    ids: List[str],
    scores: np.ndarray,
) -> List[Dict]:
    pairs = []
    for (i, j), _ in selected.items():
        if scores[i] >= scores[j]:
            a_idx, b_idx = i, j
        else:
            a_idx, b_idx = j, i
        da, db = scores[a_idx], scores[b_idx]
        pairs.append(
            {
                "a": ids[a_idx],
                "b": ids[b_idx],
                "y": 1,
                "da": float(da),
                "db": float(db),
                "d": float(abs(da - db)),
            }
        )
    return pairs


def build_split_pairs(
    df_split: pd.DataFrame,
    target_usage: int,
    min_gap: float,
    seed: int,
) -> Tuple[List[Dict], Dict]:
    ids = df_split["text_id"].astype(str).tolist()
    scores = df_split["score"].to_numpy(dtype=float)
    n = len(ids)
    if n < 2:
        return [], {"essay_count": n, "usage": {}, "per_essay_limit": 0, "target_pairs": 0, "actual_pairs": 0}

    random.seed(seed)
    bins = build_candidate_bins(scores, min_gap)
    usage = Counter({idx: 0 for idx in range(n)})
    selected: Dict[Tuple[int, int], str] = {}

    pair_phase_one(bins, usage, selected)
    target_pairs = int(round(n * target_usage / 2))
    max_cap = max(target_usage + 1, 5)
    fill_pairs(bins, usage, selected, target_pairs, max_cap)
    records = materialize_pairs(selected, ids, scores)

    stats = {
        "essay_count": n,
        "target_usage": target_usage,
        "target_pairs": target_pairs,
        "actual_pairs": len(records),
        "per_essay_limit": max_cap,
        "usage": {ids[idx]: usage[idx] for idx in range(n)},
    }
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate relative pair dataset with coverage and fixed essay usage.")
    parser.add_argument("--trait", required=True)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--target-usage", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-gap", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument(
        "--essays-csv",
        default="data/datasets/main/train_with_folds.csv",
    )
    parser.add_argument(
        "--pair-folds",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="Which folds participate in pair generation for this run.",
    )
    parser.add_argument(
        "--heldout-fold",
        default=None,
        help="Fold reserved for absolute scoring (excluded from pairing); optional metadata note.",
    )
    parser.add_argument("--output-dir", default="data/pairs_large")
    parser.add_argument("--split-fraction", type=float, default=1.0, help="Fraction of essays to keep per split (0 < f <= 1).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.essays_csv)
    trait = args.trait
    pair_folds = [fold.upper() for fold in args.pair_folds]

    pool = df[df["fold"].isin(pair_folds)][["text_id", trait, "fold"]].rename(columns={trait: "score"}).copy()
    pool["text_id"] = pool["text_id"].astype(str)
    ids = pool["text_id"].tolist()

    total = len(ids)
    test_count = int(round(total * args.test_ratio))
    val_count = int(round(total * args.val_ratio))
    train_ids, val_ids, test_ids = split_ids(ids, val_count, test_count, args.seed)

    split_frames = {
        "train": pool[pool["text_id"].isin(train_ids)].reset_index(drop=True),
        "val": pool[pool["text_id"].isin(val_ids)].reset_index(drop=True),
        "test": pool[pool["text_id"].isin(test_ids)].reset_index(drop=True),
    }

    fraction = args.split_fraction
    if not (0 < fraction <= 1):
        raise ValueError("split-fraction must be in (0, 1].")
    if fraction < 1.0:
        rng = random.Random(args.seed)
        for name, frame in list(split_frames.items()):
            if frame.empty:
                continue
            desired = max(2, int(round(len(frame) * fraction)))
            indices = list(range(len(frame)))
            rng.shuffle(indices)
            keep = indices[:desired]
            split_frames[name] = frame.iloc[keep].reset_index(drop=True)

    all_records = []
    meta = {
        "run": args.run_id,
        "trait": trait,
        "seed": args.seed,
        "target_usage": args.target_usage,
        "min_score_gap": args.min_gap,
        "pair_folds": pair_folds,
        "heldout_fold": args.heldout_fold,
        "splits": {name: frame["text_id"].tolist() for name, frame in split_frames.items()},
        "counts_requested": {},
        "counts_actual": {},
        "usage_stats": {},
    }

    for split_name, frame in split_frames.items():
        records, stats = build_split_pairs(frame, args.target_usage, args.min_gap, args.seed + hash(split_name) % 1000)
        meta["counts_requested"][split_name] = stats["target_pairs"]
        meta["counts_actual"][split_name] = stats["actual_pairs"]
        usages = stats["usage"]
        if usages:
            usage_values = list(usages.values())
            meta["usage_stats"][split_name] = {
                "per_essay_limit": stats["per_essay_limit"],
                "max_usage": max(usage_values),
                "min_usage": min(usage_values),
                "mean_usage": float(np.mean(usage_values)),
            }
        else:
            meta["usage_stats"][split_name] = {
                "per_essay_limit": 0,
                "max_usage": 0,
                "min_usage": 0,
                "mean_usage": 0.0,
            }
        all_records.extend(
            {
                "run": args.run_id,
                "trait": trait,
                "split": split_name,
                "index": idx,
                **record,
            }
            for idx, record in enumerate(records)
        )

    jsonl_path = output_dir / f"run{args.run_id}_{trait}.jsonl"
    meta_path = output_dir / f"run{args.run_id}_{trait}_meta.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record) + "\n")

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Pairs written to {jsonl_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
