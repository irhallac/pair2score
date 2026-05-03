# Reference results

Expected trait-level mean QWK values from the paper (5-fold cross-validation, co-rotated fold/seed protocol). Bold marks the best transfer setting per trait.

| Setting | Grammar | Vocabulary | Syntax |
|---------|--------:|-----------:|-------:|
| Baseline (absolute-only) | 0.6789 | 0.6140 | 0.6474 |
| Warm-start, small, standard | 0.6604 | 0.5959 | 0.6306 |
| Warm-start, small, 1-epoch | 0.6735 | 0.6152 | 0.6278 |
| Warm-start, large, standard | 0.6502 | 0.5970 | **0.6497** |
| Warm-start, large, 1-epoch | 0.6724 | 0.6000 | 0.6300 |
| Fusion, small, standard | 0.6633 | 0.5914 | 0.6271 |
| Fusion, small, 1-epoch | **0.6824** | **0.6197** | 0.6317 |
| Fusion, large, standard | 0.6611 | 0.5920 | 0.6426 |
| Fusion, large, 1-epoch | 0.6670 | 0.6019 | 0.6382 |

Each value is the mean across five held-out folds. Not all transfer configurations improve over the baseline; the paper's main finding is that transfer configuration—not just the inclusion of a pairwise stage—determines whether downstream scoring benefits. Fold-level values are reported in the paper's appendix.

## Smoke test verification

Running `configs/examples/exp00_example_smoke_pairsmini.yaml` should:

- Complete both Stage 1 and Stage 2 without errors.
- Produce `outputs/exp00_example_smoke_pairsmini/grammar/absolute_metrics_info.txt`.
- Report a QWK value (exact value will differ from the table above since the smoke config uses a minimal pair cache and single epoch).
