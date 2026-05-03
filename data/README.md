# Pairwise dataset generation

This directory holds the scripts and artefacts that feed Stage 1 (relative ranking). All numbers below come directly from the metadata files so every experiment is traceable to its data snapshot.

## Generation defaults

- Source essays: `data/datasets/main/train_with_folds.csv` (3,911 essays; folds A–E sized 789/778/785/803/756).
- Essay split per run: 80% train, 10% validation, 10% test (performed **before** pairing).
- Pairing algorithm: two-phase coverage + fill, minimum score gap |score_a − score_b| ≥ 1.0.
  1. **Coverage pass** – shuffle essays within the split and greedily form unique pairs until every essay appears at least once. Essays lacking a ≥1.0 gap partner fall back to the loosest available match so nothing is dropped.
  2. **Fill pass** – sample additional pairs by gap bucket (≥3, 2–3, 1–2) until each essay reaches the target usage. A soft per-essay cap (default 6) throttles repetition; if the pool is exhausted under the gap constraint, sampling stops early.
- Target usage: 5 appearances per essay (per split) with a soft cap of 6.
- Fold rotation: each run leaves one fold for Stage 2 evaluation (see [docs/dataset_notes.md](../docs/dataset_notes.md) for the full rotation table and seed co-rotation).

## Tooling

| Script | Purpose |
|--------|---------|
| `data/generate_pairs.py` | Low-level pair generator (see `--help`) |
| `scripts/generate_pairs_runs.sh` | Produces large + small caches for all runs/traits |
| `scripts/verify_pair_stats.py` | Prints counts, usage, and rotation info from metadata |

```bash
bash scripts/generate_pairs_runs.sh          # create large + small caches
python scripts/verify_pair_stats.py          # confirm counts and rotation
```

Small caches are produced by rerunning the generator with `split_fraction=0.5`, which retains half the essays in each split before pairing; the algorithm and parameters stay identical.

## Grammar pair statistics

| Run | Cache | Train pairs | Val pairs | Test pairs | Mean usage (train/val/test) |
|-----|-------|------------:|----------:|-----------:|:---------------------------|
| 1   | large | 6,308       | 790       | 790        | 5.00 / 5.00 / 5.00        |
| 2   | large | 6,245       | 775       | 780        | 5.00 / 5.00 / 5.00        |
| 3   | large | 6,268       | 782       | 770        | 5.00 / 5.00 / 5.00        |
| 4   | large | 6,250       | 782       | 782        | 5.00 / 5.00 / 5.00        |
| 5   | large | 6,215       | 777       | 764        | 5.00 / 5.00 / 5.00        |
| 1   | small | 3,155       | 395       | 395        | 5.00 / 5.00 / 5.00        |
| 2   | small | 3,122       | 390       | 390        | 5.00 / 5.00 / 5.00        |
| 3   | small | 3,135       | 390       | 364        | 5.00 / 5.00 / 5.00        |
| 4   | small | 3,125       | 389       | 390        | 5.00 / 5.00 / 5.00        |
| 5   | small | 3,108       | 375       | 372        | 5.00 / 5.00 / 5.00        |

## Vocabulary pair statistics

| Run | Cache | Train pairs | Val pairs | Test pairs | Mean usage (train/val/test) |
|-----|-------|------------:|----------:|-----------:|:---------------------------|
| 1   | large | 5,035       | 563       | 638        | 3.99 / 3.99 / 4.00        |
| 2   | large | 5,029       | 611       | 627        | 4.03 / 4.01 / 4.05        |
| 3   | large | 5,000       | 552       | 653        | 3.99 / 3.99 / 4.03        |
| 4   | large | 5,022       | 575       | 633        | 4.02 / 4.01 / 4.06        |
| 5   | large | 4,870       | 597       | 628        | 3.92 / 3.95 / 4.05        |
| 1   | small | 2,472       | 268       | 316        | 3.92 / 3.95 / 3.99        |
| 2   | small | 2,516       | 280       | 314        | 4.03 / 4.02 / 4.04        |
| 3   | small | 2,535       | 283       | 333        | 4.04 / 4.04 / 4.07        |
| 4   | small | 2,545       | 283       | 304        | 4.07 / 4.04 / 4.05        |
| 5   | small | 2,475       | 308       | 312        | 3.98 / 4.03 / 4.03        |

Vocabulary scores are denser, so the minimum-gap filter prunes more candidate pairs and realized mean usage stays below 5.

## Mini datasets (grammar only)

| Run | Train | Val | Test | Notes |
|-----|------:|----:|-----:|-------|
| 1   | 100   | 100 | 200  | Sampled from small run 1 pool |
| 2   | 100   | 100 | 200  | Sampled from small run 2 pool |

Mini files live in `data/pairs_mini/` for smoke tests. Regenerate from small pools if needed.
