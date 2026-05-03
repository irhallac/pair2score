# Pair2Score

Pair2Score is a two-stage framework that transfers pairwise ranking supervision into absolute scoring via parameter-efficient LLM adaptation. We evaluate on Automated Essay Scoring (AES) as an initial setting, but the formulation may generalize to other rubric-aligned or ordinal scoring tasks where comparative supervision can be derived from absolute labels.

- **Stage 1 – Relative ranking** (`src/pair2score/relative.py`): A directional Siamese LLaMA with shared LoRA adapters learns pairwise comparisons from document pairs derived from absolute trait labels, enforcing Δ(a,b) = −Δ(b,a).
- **Stage 2 – Absolute scoring** (`src/pair2score/absolute.py`): The same backbone is adapted to absolute score regression, optionally reusing Stage 1 artifacts via warm-start or embedding-fusion transfer.

## Quick start

1. Set `model.base_model` in the config to your local LLaMA checkpoint directory.
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate pair2score
   ```
   The Conda path above is the recommended and tested setup. `environment.yml` installs the PyTorch CUDA 12.1 wheels via pip, matching the environment used for our smoke tests. `requirements.txt` is provided only as a reference for manual pip setups; it is not the primary tested path.
3. Prepare the dataset (see [Dataset preparation](#dataset-preparation) below).
4. Run the smoke test:
   ```bash
   bash scripts/run_pipeline.sh configs/examples/exp00_example_smoke_pairsmini.yaml
   ```

Expected paper-level metrics (trait-level QWK) are listed in [`RESULTS.md`](RESULTS.md).

## Dataset preparation

1. Download `train.csv` from [Feedback Prize – English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) into `data/datasets/main/`. Raw essays are **not** included in this repo.
2. Inject the fold assignments:
   ```bash
   python scripts/add_folds.py \
     --input data/datasets/main/train.csv \
     --fold-map data/folds/fold_map.json \
     --output data/datasets/main/train_with_folds.csv
   ```
3. Pair caches ship with the repo (`data/pairs_small/` ≈3k pairs, `data/pairs_large/` ≈6k pairs, `data/pairs_mini/` for smoke tests). See [`data/README.md`](data/README.md) for generation details and pair statistics.

## Running experiments

Use the wrapper script, which logs each run alongside a frozen copy of its config:

```bash
bash scripts/run_pipeline.sh <CONFIG_PATH>
```

Example configurations under `configs/examples/`:

| Config | Trait | Pair cache | Stage 1 | Stage 2 |
|--------|-------|------------|---------|---------|
| `exp00_example_smoke_pairsmini` | Grammar | mini | 1 epoch | 1 epoch (smoke test) |
| `exp01_example_grammar_small_baseline` | Grammar | small | disabled | Absolute-only baseline |
| `exp02_example_grammar_small_warmstart` | Grammar | small | 10 epochs | Warm-start transfer |
| `exp03_example_vocabulary_small_fusion` | Vocabulary | small | 10 epochs | Embedding fusion |
| `exp05_example_vocabulary_large_warmstart` | Vocabulary | large | 1 epoch | Warm-start transfer |
| `exp06_example_syntax_large_fusion` | Syntax | large | 10 epochs | Embedding fusion |

## Model overview

- **Backbone:** LLaMA-3.2-1B, loaded from a local checkpoint referenced via `model.base_model` in each config.
- **Stage 1 (Siamese):** Both documents share one backbone + LoRA adapter (r=16, α=32, dropout 0.05 on q/k/v/o). A bias-free linear utility head produces scalar utilities whose difference serves as the comparison logit.
- **Stage 2 (Transfer):** Warm-start initializes the absolute-stage adapter from Stage 1; fusion additionally concatenates a frozen Stage 1 embedding with the current pooled representation. A baseline variant trains from scratch without Stage 1.
- **Pipeline:** `scripts/run_pipeline.sh` runs Stage 1 (if enabled) then Stage 2, storing logs and config snapshots under `outputs/` and checkpoints under `checkpoints/`.

## Outputs

Each run produces:
- `outputs/<experiment>/<trait>/run_*.log` – console log
- `outputs/<experiment>/<trait>/run_*_config.yaml` – frozen config snapshot
- `outputs/<experiment>/<trait>/absolute_metrics_info.txt` – summary metrics (QWK, MAE)
- `checkpoints/<experiment>/<trait>/relative/` – Stage 1 adapter, head, embeddings (when enabled)
- `checkpoints/<experiment>/<trait>/absolute/` – Stage 2 checkpoints

## Environment

- Python 3.10+, PyTorch 2.4 (CUDA 12.1 wheels), CUDA 12.1+
- GPU with ≥16 GB memory (LLaMA-3.2-1B + LoRA)
- Dependencies: `conda env create -f environment.yml` (recommended, tested). `requirements.txt` lists the pinned Python packages for manual setups, but our end-to-end smoke test used the Conda route above.
- LLaMA checkpoint: download the model (after accepting the license) from `https://huggingface.co/meta-llama/Llama-3.2-1B` and set `model.base_model` in configs.

## Further documentation

- Pair generation details and statistics: [`data/README.md`](data/README.md)
- Reproducibility guide: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- Stage 1 architecture reference: [`docs/siamese_llama_reference.md`](docs/siamese_llama_reference.md)
- Dataset notes and fold rotation: [`docs/dataset_notes.md`](docs/dataset_notes.md)

## Licenses

- **Code:** MIT License (see `LICENSE`)
- **Models:** LLaMA weights are not distributed here; obtain them from `https://huggingface.co/meta-llama/Llama-3.2-1B` under the LLaMA license.
- **Data:** Feedback Prize – English Language Learning dataset from Kaggle; follow the competition's terms of use.

## Citation

Citation details will be added upon publication.
