# Pair2Score

Pair2Score is a two-stage framework that leverages pairwise supervision to improve any document/content scoring task (sentiment intensity, readability, toxicity, image aesthetics, medical severity, LLM response grading, etc.). Automatic Essay Scoring (AES) is simply our first sandbox: we iterate here so the transfer ideas can mature into a general technique.

This repository is a work-in-progress research prototype; APIs and configs may evolve as we refine the method and extend it beyond AES.

The name reflects the core idea—**pair → score**:
- **pair** – Stage 1 learns from pairs of documents (A vs B), predicting which one is better.
- **2** – denotes the transition from relative judgments to absolute predictions.
- **score** – Stage 2 outputs the final numeric score for each document.

1. **Stage 1 – Relative ranking (`src/pair2score/relative.py`).** Train a Siamese LLaMA with LoRA adapters on cached pairwise comparisons so Δ(a,b)=−Δ(b,a). This stage captures preference signals that are often easier or cheaper to annotate.
2. **Stage 2 – Absolute scoring (`src/pair2score/absolute.py`).** Adapt the same backbone to the downstream regression/classification objective, optionally reusing or fusing the Stage 1 adapters/embeddings. The goal is to transfer as much of the relative knowledge as possible; the warm-start / stacking / fusion variants in this repo are our first step, and we plan to keep expanding this transfer space.

## Environment

- Python 3.10+
- CUDA-capable GPU with enough memory for LLaMA‑3.2‑1B + LoRA heads.
- Install dependencies into your environment:
  ```bash
  pip install torch transformers peft pandas tqdm
  ```

The repo assumes documents (essays in our AES runs) live at `data/datasets/main/train_with_folds.csv` and pair caches under `data/`.  
**Backbone requirement:** Download a LLaMA checkpoint from Hugging Face (after accepting its license), store it locally, and point the model path in each config to that directory (the loader expects a folder with `model/` and `tokenizer/` subdirectories or an equivalent flat layout).

## Dataset prep

1. Download `train.csv` from [Feedback Prize – English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) into `data/datasets/main/`.
2. Inject the official folds with the helper script:
   ```bash
   python scripts/add_folds.py \
     --input data/datasets/main/train.csv \
     --fold-map data/folds/fold_map.json \
     --output data/datasets/main/train_with_folds.csv
   ```
3. Pair caches (`data/pairs_small/*.jsonl`, `data/pairs_large/*.jsonl`) plus a micro `data/pairs_mini/` set ship with the repo; regenerate them only if you want to rebuild from scratch.

## Running the pipeline

Use the wrapper script so each run is logged alongside a frozen copy of its config:

```bash
bash scripts/run_pipeline.sh <CONFIG_PATH>
```

Representative YAMLs live under `configs/examples/` so you can copy or adapt them:

| File | Trait / Cache | Stage 1 | Stage 2 note |
|------|---------------|---------|--------------|
| `exp00_example_smoke_pairsmini.yaml` | Grammar / `pairs_mini` | 1-epoch sanity run | 1 absolute epoch; fastest smoke test |
| `exp01_example_grammar_small_baseline.yaml` | Grammar / small | disabled | Absolute-only baseline |
| `exp02_example_grammar_small_warmstart.yaml` | Grammar / small | 10 epochs, adapter reused | Warm-start absolute scorer |
| `exp03_example_vocabulary_small_fusion.yaml` | Vocabulary / small | 10 epochs, adapter reused | Embedding fusion enabled |
| `exp05_example_vocabulary_large_warmstart.yaml` | Vocabulary / large | 1 epoch, adapter reused | Large-cache warm-start |
| `exp06_example_syntax_large_fusion.yaml` | Syntax / large | 10 epochs, adapter reused | Fusion variant on syntax |

Run them directly or copy into `configs/` if you want to keep your experiments separate:

```bash
# Fast smoke test (≈1 epoch per stage)
bash scripts/run_pipeline.sh configs/examples/exp00_example_smoke_pairsmini.yaml

# Canonical grammar baseline / warm-start / fusion examples
bash scripts/run_pipeline.sh configs/examples/exp01_example_grammar_small_baseline.yaml
bash scripts/run_pipeline.sh configs/examples/exp02_example_grammar_small_warmstart.yaml

# Vocabulary + Syntax variants
bash scripts/run_pipeline.sh configs/examples/exp03_example_vocabulary_small_fusion.yaml
bash scripts/run_pipeline.sh configs/examples/exp05_example_vocabulary_large_warmstart.yaml
bash scripts/run_pipeline.sh configs/examples/exp06_example_syntax_large_fusion.yaml
```

## Model overview
- **Backbone (LLM / LLaMA)** – All experiments in this repo use a LLaMA-based LLM checkpoint downloaded from Hugging Face and stored locally; configs reference it via the `model.base_model` path. The loader understands both canonical layouts (`config.json` at root or `model/` + `tokenizer/` folders).
- **Siamese Stage 1** – Both inputs flow through one tokenizer/backbone stack outfitted with LoRA adapters (defaults: r=16, α=32, dropout=0.05 on q/k/v/o), so the two utilities are produced by identical weights. The bias-free relative head then compares those utilities, yielding Δ(a,b)=−Δ(b,a) and enforcing directional ranking.
- **Stage 2 reuse** – Absolute scoring can warm-start from the Stage 1 adapters, stack fresh ones, or run a pure baseline without LoRA. Fusion configs also reuse the cached embeddings produced during Stage 1. These are our first transfer variants—this repo is the playground where we keep extending how Stage 1 knowledge flows into Stage 2.
- **Pipeline flow** – `scripts/run_pipeline.sh` takes one YAML, runs Stage 1 if enabled, then Stage 2, storing logs/config snapshots under `outputs/` and checkpoints under `checkpoints/`.

### Small pair cache experiments (default setting)

The relative stage always runs the Siamese LLaMA described above (two towers sharing weights and a bias-free head), so every small/large cache experiment differs only by the amount of Stage 1 data and how the adapters are reused in Stage 2.

The small cache (`data/pairs_small/run1_grammar.jsonl`) contains ~3.1 k train pairs.

### Large pair cache experiments

The large cache (`data/pairs_large/run1_grammar.jsonl`) doubles the pair counts and mirrors the small-run hyperparameters.

## Outputs

Each run produces:
- `outputs/<experiment>/<trait>/run_*.log` – console log.
- `outputs/<experiment>/<trait>/run_*_config.yaml` – frozen config snapshot.
- `outputs/<experiment>/<trait>/absolute_metrics_info.txt` – summary metrics (MAE, QWK, etc.).
- `checkpoints/<experiment>/<trait>/relative/` – Stage 1 adapter/head/embeddings when enabled.
- `checkpoints/<experiment>/<trait>/absolute/` (or top-level files) – Stage 2 checkpoints.

## Documentation

- Pair generation details: [data/README.md](data/README.md)
- Reproducibility guide: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- Stage 1 architecture details: [docs/siamese_llama_reference.md](docs/siamese_llama_reference.md)

Raw Kaggle essays are **not** included in this repo. Use `scripts/add_folds.py` with your local `train.csv` to recreate `data/datasets/main/train_with_folds.csv` before launching experiments.

## Licenses

- **Code** – Released under the MIT License (see `LICENSE`).
- **Models** – LLaMA weights are **not** distributed here; obtain them from Hugging Face under the LLaMA license.
- **Data** – Experiments use the Feedback Prize – English Language Learning dataset from Kaggle. Follow the competition’s terms of use when downloading and using `train.csv` .

## How to cite

Citation details will be added once the Pair2Score paper is published.
