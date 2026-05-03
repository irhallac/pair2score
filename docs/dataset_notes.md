# Dataset notes

This document explains the experimental design behind the pair caches and fold rotation. For exact pair counts and generation scripts, see [`data/README.md`](../data/README.md).

## Source data

Experiments use the **Feedback Prize – English Language Learning** dataset (Kaggle). We work with three analytic traits—grammar, vocabulary, and syntax—each treated as an independent regression target with scores in [1.0, 5.0] at 0.5 increments.

Raw CSV: `data/datasets/main/train_with_folds.csv` (3,911 essays; folds A–E sized 789/778/785/803/756).

## Fold rotation and seed co-rotation

Each run holds out one fold for Stage 2 evaluation. The training seed co-rotates with the held-out fold so that stochasticity is part of the measured evidence.

| Run | Held-out fold | Training folds | Seed |
|-----|---------------|----------------|------|
| 1   | E             | A, B, C, D     | 36   |
| 2   | A             | B, C, D, E     | 42   |
| 3   | B             | C, D, E, A     | 48   |
| 4   | C             | D, E, A, B     | 54   |
| 5   | D             | E, A, B, C     | 60   |

Stage 2 always evaluates on the held-out fold. Stage 1 pair caches must match the same run ID so both stages see identical essay splits.

## Pair generation logic

Pair caches live under `data/pairs_small/` and `data/pairs_large/`. The generation procedure (implemented in `data/generate_pairs.py`):

1. **Split before pairing:** Within each run, training-fold essays are split 80/10/10 into Stage 1 train/val/test pools. Stage 2 receives the full folds directly.
2. **Nominal score-gap filter:** Pairing targets essays with absolute score difference ≥ 1.0.
3. **Coverage pass:** Greedily form pairs until every essay appears at least once. When an essay has no available partner under the nominal threshold, the coverage step can fall back to smaller realized gaps, including occasional ties, to preserve coverage.
4. **Fill pass:** Sample additional pairs by gap bucket (≥3, 2–3, 1–2) until each essay reaches the target usage (default 5, soft-capped at 6).
5. **Cache sizes:**
   - **Large:** Full essay pool from training folds (≈6k train pairs per trait/run).
   - **Small:** 50% subsample of essays before pairing (≈3k train pairs), same algorithm.
   - **Mini:** Tiny samples from small pools (100/100/200 pairs) for smoke tests, grammar only.

## Baseline note

For absolute-only baselines (`stage1_relative.enabled: false`), Stage 2 runs once per trait/run. The baseline QWK is identical regardless of pair-cache size since Stage 1 is disabled and Stage 2 data is unchanged.

## Why this matters

Stage 1 comparisons drive the relative-to-absolute transfer. Knowing the exact pair coverage and fold alignment helps interpret which configurations benefit from warm-start or fusion. Every run's Stage 1/Stage 2 setup is traceable through the metadata files shipped with each pair cache.
