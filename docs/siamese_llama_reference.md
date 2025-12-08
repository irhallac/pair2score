# Siamese LLaMA Reference

Authoritative description of the Stage 1 model implemented in src/pair2score/relative.py.

## Concept
- Backbone: LLaMA‑3.2‑1B LLM (base) with LoRA adapters (default r=16, alpha=32).
- Topology: Siamese — inputs A and B share tokenizer, encoder, and adapters.
- Utilities: linear head (bias=False) maps pooled embeddings to scalar utilities s(h).
- Directional ranking: logits Δ = s(hₐ) − s(h_b), so Δ(a,b) = −Δ(b,a).
- Loss: BCEWithLogitsLoss(Δ, y), where y=1 if A should rank higher than B.
- Re-use: utilities s(h) seed Stage 2 (warm-start or embedding fusion).

## Pseudocode
```
emb_a = encode(backbone, essay_a)
emb_b = encode(backbone, essay_b)
score_a = scorer(emb_a)
score_b = scorer(emb_b)
Δ = score_a - score_b  # antisymmetric
loss = BCEWithLogitsLoss(Δ, label)
```

## Architecture (ASCII Overview)
```
            ┌────────────────────┐       ┌────────────────────┐
            │    Essay Text A    │       │    Essay Text B    │
            └────────┬───────────┘       └────────┬───────────┘
                     │                            │
            ┌────────▼────────┐          ┌────────▼────────┐
            │ Tokenizer (LLM) │          │ Tokenizer (LLM) │
            └────────┬────────┘          └────────┬────────┘
                     │                            │
            ┌────────▼────────┐          ┌────────▼────────┐
            │   LLaMA Encoder │◄────────►│   LLaMA Encoder │
            │ (shared weights)│          │ (shared weights)│
            └────────┬────────┘          └────────┬────────┘
                     ▼                            ▼
            ┌────────────────────┐     ┌────────────────────┐
            │ Mean Pool + Mask   │     │ Mean Pool + Mask   │
            └────────┬───────────┘     └────────┬───────────┘
                     ▼                            ▼
                 s(a) = Linear(h_a)           s(b) = Linear(h_b)
                     └───────────────┬───────────────┘
                                     ▼
                          Δ = s(a) − s(b)
                                     │
                           BCEWithLogitsLoss
```

## Properties to cite
1. Enforces directional ranking (A>B ≠ B>A).
2. Antisymmetric logits via subtraction and bias-free head.
3. Same mathematical footing as RankNet / Bradley–Terry–Luce.
4. Lightweight updates (~3M trainable params for r=16).
5. Utilities transferable to absolute stage.

## Sanity check snippet
```python
logit_ab = model(emb_a, emb_b)
logit_ba = model(emb_b, emb_a)
assert torch.allclose(logit_ab, -logit_ba, atol=1e-6)
```
