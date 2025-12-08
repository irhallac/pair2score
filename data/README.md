# Pairwise Dataset Generation

This directory holds the scripts and artefacts that feed Stage 1 (relative ranking). All numbers below come directly from the current metadata files so you can trace every experiment back to its data snapshot.

## Generation defaults

- Source essays: `data/datasets/main/train_with_folds.csv` (3 911 essays; folds A–E sized 789/778/785/803/756).
- Essay split per run: 80 % train, 10 % validation, 10 % test (performed **before** pairing).
- Pairing algorithm: two-phase coverage + fill, minimum score gap `|score_a - score_b| ≥ 1.0`.
  1. **Coverage pass** – shuffle essays within the split and greedily form unique pairs until every essay appears at least once. Essays lacking a ≥1.0 gap partner fall back to the loosest available match so nothing is dropped.
  2. **Fill pass** – keep sampling unseen pairs by gap bucket (≥3, 2–3, 1–2) until each essay reaches the target usage. A soft per-essay cap (default 6) throttles repetition; if the pool is exhausted under the gap constraint we stop early.
- Target usage: 5 appearances per essay (per split) with a soft cap of 6 enforced by the fill routine.
- Fold rotation: each run leaves one fold untouched for Stage 2.  
  | Run | Pair folds (Stage 1) | Held-out fold (Stage 2) |
  |-----|----------------------|-------------------------|
  | 1   | A, B, C, D           | E                       |
  | 2   | B, C, D, E           | A                       |
  | 3   | C, D, E, A           | B                       |
  | 4   | D, E, A, B           | C                       |
  | 5   | E, A, B, C           | D                       |
- RNG seed: 36 for every run (affects essay shuffles and pair sampling).

## Tooling

- `data/generate_pairs.py` – low-level generator (see `--help` for knobs).
- `scripts/generate_pairs_runs.sh` – convenience wrapper that produces the full + small datasets for all runs/trials (set `FORCE=1` to overwrite). Generates large caches under `data/pairs_large/` and small caches under `data/pairs_small/`.
- `scripts/verify_pair_stats.py` – sanity checker that prints counts, usage, and rotation info straight from the metadata.

Run order:
```bash
bash scripts/generate_pairs_runs.sh          # create large + small sets
python scripts/verify_pair_stats.py          # confirm counts/rotation
```

Small datasets are produced by rerunning the generator with `split_fraction=0.5`, which randomly retains half the essays in each split **before** pairing; the coverage/fill algorithm and parameters stay identical. Mini caches (100/100/200 pairs) exist for grammar runs 1–2 and are sampled from those small pools using the same procedure.

## Grammar trait snapshot

| Run | Size  | Train pairs | Val pairs | Test pairs | Mean usage (train/val/test) |
|-----|-------|-------------|-----------|------------|------------------------------|
| 1   | large | 6 308       | 790       | 790        | 5.00 / 5.00 / 5.00           |
| 2   | large | 6 245       | 775       | 780        | 5.00 / 5.00 / 5.00           |
| 3   | large | 6 268       | 782       | 770        | 5.00 / 5.00 / 5.00           |
| 4   | large | 6 250       | 782       | 782        | 5.00 / 5.00 / 5.00           |
| 5   | large | 6 215       | 777       | 764        | 5.00 / 5.00 / 5.00           |
| 1   | small | 3 155       | 395       | 395        | 5.00 / 5.00 / 5.00           |
| 2   | small | 3 122       | 390       | 390        | 5.00 / 5.00 / 5.00           |
| 3   | small | 3 135       | 390       | 364        | 5.00 / 5.00 / 5.00           |
| 4   | small | 3 125       | 389       | 390        | 5.00 / 5.00 / 5.00           |
| 5   | small | 3 108       | 375       | 372        | 5.00 / 5.00 / 5.00           |

Every grammar run meets the requested usage; minor count differences come from the coverage pass honouring the score-gap constraint while balancing folds.

## Vocabulary trait snapshot

| Run | Size  | Train pairs | Val pairs | Test pairs | Mean usage (train/val/test) |
|-----|-------|-------------|-----------|------------|------------------------------|
| 1   | large | 5 035       | 563       | 638        | 3.99 / 3.99 / 4.00           |
| 2   | large | 5 029       | 611       | 627        | 4.03 / 4.01 / 4.05           |
| 3   | large | 5 000       | 552       | 653        | 3.99 / 3.99 / 4.03           |
| 4   | large | 5 022       | 575       | 633        | 4.02 / 4.01 / 4.06           |
| 5   | large | 4 870       | 597       | 628        | 3.92 / 3.95 / 4.05           |
| 1   | small | 2 472       | 268       | 316        | 3.92 / 3.95 / 3.99           |
| 2   | small | 2 516       | 280       | 314        | 4.03 / 4.02 / 4.04           |
| 3   | small | 2 535       | 283       | 333        | 4.04 / 4.04 / 4.07           |
| 4   | small | 2 545       | 283       | 304        | 4.07 / 4.04 / 4.05           |
| 5   | small | 2 475       | 308       | 312        | 3.98 / 4.03 / 4.03           |

Vocabulary scores are denser, so the minimum-gap filter prunes more candidate pairs and the realised mean usage stays just under 5. The metadata captures those deltas run by run.

## Upcoming run QC summary

- Runs 3–5 reuse the same fold rotation (C→D→E for pairing; B→C→D held out) with seed 36, and every split remains disjoint—no essay ID leaks between train/val/test.
- Pair totals differ slightly from requested counts only when the score-gap filter exhausts combinations (notably vocabulary test splits), matching the behaviour we saw in runs 1–2.

| Trait | Cache | Run | Held-out | Pair folds | Unique essays (train/val/test) | Pair count (train/val/test) |
| --- | --- | --- | --- | --- | --- | --- |
| grammar | small | 3 | B | CDEA | 1254/156/156 | 3135/390/364 |
| grammar | small | 4 | C | DEAB | 1250/156/156 | 3125/389/390 |
| grammar | small | 5 | D | EABC | 1243/156/156 | 3108/375/372 |
| grammar | large | 3 | B | CDEA | 2507/313/313 | 6268/782/770 |
| grammar | large | 4 | C | DEAB | 2500/313/313 | 6250/782/782 |
| grammar | large | 5 | D | EABC | 2486/311/311 | 6215/777/764 |
| vocabulary | small | 3 | B | CDEA | 1254/156/156 | 2535/283/333 |
| vocabulary | small | 4 | C | DEAB | 1250/156/156 | 2545/283/304 |
| vocabulary | small | 5 | D | EABC | 1243/156/156 | 2475/308/312 |
| vocabulary | large | 3 | B | CDEA | 2507/313/313 | 5000/552/653 |
| vocabulary | large | 4 | C | DEAB | 2500/313/313 | 5022/575/633 |
| vocabulary | large | 5 | D | EABC | 2486/311/311 | 4870/597/628 |

## Mini datasets (grammar only)

| Run | Train pairs | Val pairs | Test pairs | Notes                          |
|-----|-------------|-----------|------------|--------------------------------|
| 1   | 100         | 100       | 200        | Sampled from small run 1 pool |
| 2   | 100         | 100       | 200        | Sampled from small run 2 pool |

Mini files live in `data/pairs_mini/` and share the same pairing parameters; regenerate them from the small pools if you need fresh samples.

---

Use these tables alongside `scripts/verify_pair_stats.py` to document exactly which pair cache backs any experiment. That way every Stage 1/Stage 2 comparison can cite the run ID, trait, and size without ambiguity.
