# Pair2Score Dataset Notes

This document accompanies the exported summaries in `docs/` so readers understand how the Stage 1 pair datasets are built and how the Stage 2 splits stay aligned.

---

## Source data, model, & absolute scoring task
- The current experiments use the **Feedback Prize – English Language Learning** dataset (Kaggle competition; six analytic traits per essay with scores on trait-specific scales). We work with the grammar, vocabulary, and syntax traits—each is treated as an independent regression target.
- Relative Stage 1 uses a **Siamese LLaMA LoRA adapter** on top of an LLM backbone: both towers share the same LLaMA backbone, and a lightweight linear head predicts the score difference (`score_a – score_b`). Only the LoRA adapters plus head are trained in Stage 1.
- Stage 2 (absolute scoring) fine‑tunes the absolute regressor on the four non‑held‑out folds and evaluates on the remaining fold, optionally reusing the Stage 1 adapters (warm‑start) or fusing Stage 1 embeddings (fusion). Because Stage 1 enforces Δ(a,b)=−Δ(b,a), the directional ranking behavior is preserved even if Stage 2 trains from scratch.
- Raw CSV: `data/datasets/main/train_with_folds.csv`, a curated version of the Kaggle training set with five stratified folds (A–E). In this repo we instantiate the absolute task as Automated Essay Scoring (AES): predict the trait score for each essay; we report mean absolute error (MAE) and quadratic weighted kappa (QWK) on the held‑out fold. The same structure can be reused for other document‑level scoring tasks.

## Fold rotation for Stage 1 & Stage 2
For each run we rotate which fold is held out:

| Run | Held-out fold (`test_fold`) | Training folds | Stage seed |
|-----|-----------------------------|----------------|------------|
| 1   | E                           | A, B, C, D     | 36 |
| 2   | A                           | B, C, D, E     | 42 |
| 3   | B                           | C, D, E, A     | 48 |
| 4   | C                           | D, E, A, B     | 54 |
| 5   | D                           | E, A, B, C     | 60 |

Stage 2 always uses the held-out fold for testing. Stage 1 (relative) must load the pair cache that matches the same run id so both stages see identical essay splits.

---

## Pair generation logic

Pair caches live under `data/pairs_small/` and `data/pairs_large/`. Every run/trait/cache combo has two files: a JSONL of pairs and a metadata JSON. Here is the logic implemented in `data/generate_pairs.py`:

1. **Split essays before pairing**: within each run, essays from the training folds are split 80/10/10 into train/val/test pools for Stage 1. (Stage 2 receives the full folds directly.)
2. **Score-gap filter**: only pair essays whose absolute score difference ≥ 1.0. This keeps relative comparisons meaningful.
3. **Coverage pass**: shuffle essays within each split and greedily form unique pairs until every essay appears at least once. If an essay cannot find a ≥ 1.0 gap partner, we relax the gap progressively (down to 0.5) so it is not dropped.
4. **Fill pass**: continue sampling unseen pairs bucketed by gap (≥3, 2–3, 1–2) until each essay reaches the target usage (default 5 appearances per split, soft-capped at 6). Sampling is random but seeded, so reruns reproduce the same pair lists.
5. **Small versus large caches**:
   - Large caches use the full essay pool from the training folds (≈6 k train pairs per trait/run).
   - Small caches rerun the generator with `split_fraction=0.5` to down-sample essays before pairing, yielding ~2.5 k train pairs with the same algorithm.
   - Mini caches (grammar only) are tiny samples from the small pools (100/100/200 pairs) for smoke tests.

All metadata (pair counts, usage histograms, fold membership) is stored in the accompanying `_meta.json` files. `scripts/verify_pair_stats.py` prints those stats so every experiment can cite its exact pair backing. This matters because Stage 1 (relative) pretraining feeds Stage 2 (absolute) via warm-start or fusion: if the pair cache changes, the absolute scores may drift. For absolute-only baselines (`stage1_relative.enabled: false`) we run Stage 2 once per trait/run (using the small cache config) and reuse the identical metrics for the “large” baseline column since Stage 1 is disabled and the data entering Stage 2 is the same.

---

## Quick reference tables

### Grammar
| Run | Cache | Train pairs | Val pairs | Test pairs | Mean usage (train/val/test) |
|-----|-------|-------------|-----------|------------|------------------------------|
| 1   | large | 6 308 | 790 | 790 | 5.00 / 5.00 / 5.00 |
| 2   | large | 6 245 | 775 | 780 | 5.00 / 5.00 / 5.00 |
| 3   | large | 6 268 | 782 | 770 | 5.00 / 5.00 / 5.00 |
| 4   | large | 6 250 | 782 | 782 | 5.00 / 5.00 / 5.00 |
| 5   | large | 6 215 | 777 | 764 | 5.00 / 5.00 / 5.00 |
| 1   | small | 3 155 | 395 | 395 | 5.00 / 5.00 / 5.00 |
| 2   | small | 3 122 | 390 | 390 | 5.00 / 5.00 / 5.00 |
| 3   | small | 3 135 | 390 | 364 | 5.00 / 5.00 / 5.00 |
| 4   | small | 3 125 | 389 | 390 | 5.00 / 5.00 / 5.00 |
| 5   | small | 3 108 | 375 | 372 | 5.00 / 5.00 / 5.00 |

### Vocabulary
| Run | Cache | Train pairs | Val pairs | Test pairs | Mean usage (train/val/test) |
|-----|-------|-------------|-----------|------------|------------------------------|
| 1   | large | 5 035 | 563 | 638 | 3.99 / 3.99 / 4.00 |
| 2   | large | 5 029 | 611 | 627 | 4.03 / 4.01 / 4.05 |
| 3   | large | 5 000 | 552 | 653 | 3.99 / 3.99 / 4.03 |
| 4   | large | 5 022 | 575 | 633 | 4.02 / 4.01 / 4.06 |
| 5   | large | 4 870 | 597 | 628 | 3.92 / 3.95 / 4.05 |
| 1   | small | 2 472 | 268 | 316 | 3.92 / 3.95 / 3.99 |
| 2   | small | 2 516 | 280 | 314 | 4.03 / 4.02 / 4.04 |
| 3   | small | 2 535 | 283 | 333 | 4.04 / 4.04 / 4.07 |
| 4   | small | 2 545 | 283 | 304 | 4.07 / 4.04 / 4.05 |
| 5   | small | 2 475 | 308 | 312 | 3.98 / 4.03 / 4.03 |

### Syntax
(Seed 36 pairs generated later but follow the same statistics; metadata lives alongside the trait-specific files.)

---

## Why this matters
- Stage 1 comparisons drive the relative → absolute transfer. Knowing the exact pair coverage and usage helps interpret which runs benefit most from warm-start/fusion.
- Fold alignment ensures we never leak essays between relative and absolute evaluations; every run’s Stage 1/Stage 2 setup is traceable through the tables above.

Use this file together with the dated summaries when sharing results externally; it captures the “business logic” of the data generation without exposing internal scripts.
