# Reproducibility Guide

This document gives a concrete recipe for recreating the experiments in this repository. It assumes you have already skimmed the top‑level `README.md` for the high‑level idea.

## 1. Environment

Create the Conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate pair2score
```

This Conda recipe is the tested installation path for this public release. It uses Python 3.10 with:
- PyTorch 2.4.0 (CUDA 12.1)
- transformers 4.57.1
- peft 0.15.0
- pandas 2.2.3
- scikit‑learn 1.5.2
- tqdm 4.66.5
- PyYAML 6.0.2

Internally, `environment.yml` installs the CUDA 12.1 PyTorch wheels from the official PyTorch wheel index via pip, matching the environment used for our smoke-tested runs.

We ran on GPUs with ≥16 GB VRAM (A100 / RTX8000 / H100). Smaller cards may work with reduced batch sizes but are not tested.

## 2. Model checkpoint (LLaMA)

We do **not** ship any LLaMA weights. To run the code:

1. Accept the LLaMA license and download the checkpoint you plan to use from the official Hugging Face page, for example `https://huggingface.co/meta-llama/Llama-3.2-1B`.
2. Store the model anywhere on your filesystem. The loader understands both layouts:
   - `.../config.json` and `.../tokenizer_config.json` at the root, or
   - `.../model/config.json` and `.../tokenizer/tokenizer_config.json`.
3. In each config you run, set the `base_model` path to that directory, for example:
   ```yaml
   model:
     base_model: /path/to/local/meta-llama-checkpoint
   ```

The directory above must exist locally before you launch any experiments; all of our runs referenced a local LLaMA checkpoint this way.

## 3. Data preparation

1. Accept the Kaggle  
   [Feedback Prize – English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)  
   terms and download `train.csv`:
   ```bash
   kaggle competitions download -c feedback-prize-english-language-learning -p data/datasets/main
   unzip -d data/datasets/main data/datasets/main/feedback-prize-english-language-learning.zip
   ```
   We do **not** redistribute this file.
2. Use the provided `data/folds/fold_map.json` and helper script to augment your local `train.csv` and produce `data/datasets/main/train_with_folds.csv` (Kaggle data + a `fold` column A–E):
   ```bash
   python scripts/add_folds.py \
     --input data/datasets/main/train.csv \
     --fold-map data/folds/fold_map.json \
     --output data/datasets/main/train_with_folds.csv
   ```
3. Pair caches (`data/pairs_small/*.jsonl`, `data/pairs_large/*.jsonl`, and `data/pairs_mini/*.jsonl`) are already included in the repo. Only run the generator if you need to rebuild them:
   ```bash
   bash scripts/generate_pairs_runs.sh
   python scripts/verify_pair_stats.py
   ```

## 4. Running the pipeline

1. Choose a config under `configs/examples/` that matches the variant you want:

   | File | Trait | Cache | Stage 1 | Stage 2 highlight |
   |------|-------|-------|---------|-------------------|
   | `exp00_example_smoke_pairsmini.yaml` | grammar | `pairs_mini` | 1 epoch | 1‑epoch absolute sanity run |
   | `exp01_example_grammar_small_baseline.yaml` | grammar | small (`run2_grammar`) | disabled | Absolute‑only baseline |
   | `exp02_example_grammar_small_warmstart.yaml` | grammar | small (`run3_grammar`) | 10 epochs, reuse adapter | Warm‑start absolute model |
   | `exp03_example_vocabulary_small_fusion.yaml` | vocabulary | small (`run4_vocabulary`) | 10 epochs, reuse adapter | Embedding fusion enabled |
   | `exp05_example_vocabulary_large_warmstart.yaml` | vocabulary | large (`run2_vocabulary`) | 1 epoch, reuse adapter | Large‑cache warm‑start |
   | `exp06_example_syntax_large_fusion.yaml` | syntax | large (`run3_syntax`) | 10 epochs, reuse adapter | Fusion variant on syntax |

   Copy a YAML and adjust:
   - `model.base_model` → your local LLaMA directory.
   - `hardware.cuda_device` → GPU index to use.
   - Optional: batch sizes / epochs if you need shorter runs.

2. Launch both stages (Stage 1 relative + Stage 2 absolute) with a single command from the repo root:
   ```bash
   bash scripts/run_pipeline.sh configs/<experiment>.yaml
   ```

3. For a quick sanity check, run the mini‑pair smoke test:
   ```bash
   bash scripts/run_pipeline.sh configs/examples/exp00_example_smoke_pairsmini.yaml
   ```
   This smoke config is the recommended first run after setting `model.base_model` and preparing `train_with_folds.csv`.

## 5. Outputs and verification

Each run writes into:

- `outputs/expXX_*/*/run_YYYY-MM-DD_HH-MM-SS.log`  
  Contains both Stage 1 `[relative]` and Stage 2 `[absolute]` logs.
- `outputs/expXX_*/*/run_*_config.yaml`  
  Frozen copy of the config used for that run.
- `outputs/expXX_*/*/absolute_metrics_info.txt`  
  MAE/QWK for the best absolute‑stage epoch.
- `checkpoints/expXX_*/*/relative/`  
  Stage 1 adapter, relative head, and optional embeddings.
- `checkpoints/expXX_*/*/absolute_run_info.txt`  
  Compact summary of best‑epoch metrics for Stage 2.

To confirm a run:

1. Open the log and look for a final `[absolute] finished` line with train/val/test MAE and QWK.
2. Inspect the corresponding `absolute_run_info.txt` to double‑check the best epoch and test metrics.

For architectural details of the Siamese Stage 1 model, see `docs/siamese_llama_reference.md`.
