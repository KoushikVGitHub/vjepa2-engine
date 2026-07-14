# vjepa2-engine

**A from-scratch, dataset-agnostic engineering pipeline for self-supervised world-models:** data curation at scale → distributed training with principled anti-collapse (LeJEPA / SIGReg) → optimized inference. The three things a world-model lab runs every day, built end to end and benchmarked with logged numbers.

> Self-supervised models live or die on infrastructure: how fast you can curate the data, how efficiently you shard a model across GPUs without it collapsing, and how cheaply you can serve it. This repo builds and benchmarks that engineering surface. The engine is **dataset-agnostic** — the loss, the distributed SIGReg, and the FSDP trainer don't change with the data; only the loader and input dims do.

It is currently being proven on **CAMELS cosmological field maps** (a scientific world-model task with a clean quantitative benchmark), and escalates from there toward true spatiotemporal data — 3D fields across cosmic time, then observed solar video (see [Roadmap](#roadmap)).

Built as a focused engineering intensive. Method credibility (from-scratch JEPA + modern anti-collapse) is included so the training loop isn't a black box, but the headline is **systems, throughput, and cost** — not a new model.

## The engine (dataset-agnostic)

| Stage | File | What it does | Status |
|---|---|---|---|
| **1 · Curation at scale** | `src/data/fields.py`, `src/data/curation.py` | Offline, manifest-driven curation with disk-cached stats (skip the full re-scan on reruns); standardization computed in one pass. Profiled throughput and found the bottleneck *moves* under parallelism. | ✅ built + profiled |
| **2 · The method (from scratch)** | `src/jepa_loss.py`, `src/sigreg.py` | Minimal JEPA (masking, predictor) with two anti-collapse paths: EMA + stop-grad, and **LeJEPA / SIGReg** (push embeddings toward an isotropic Gaussian — no EMA/stop-grad heuristics). Distributed SIGReg all-reduce **verified** (world=2 ≡ world=1). Effective-rank monitoring to catch dimensional collapse. | ✅ built + verified |
| **3 · Distributed training** | `src/train_fsdp.py` | Wrap the JEPA loop in **PyTorch FSDP/DDP**, bf16 mixed precision, activation checkpointing; LR warmup + cosine + grad-clip for stable LeJEPA. Throughput + memory + MFU logged per lever. | ✅ built + benchmarked |
| **4 · Inference optimization** | `src/infer.py` | Baseline encoder inference → `torch.compile`, bf16/fp16, batching, optional int8 PTQ. Latency (p50/p99) + throughput benchmark, lever-by-lever. | 🔜 next |

## Current application — Stage 1: CAMELS cosmology fields

[CAMELS](https://camels.readthedocs.io) provides 2D maps of 13 physical fields (gas/dark-matter/stellar density, temperature, HI, velocity, …) from thousands of simulated universes. The task: recover the **cosmological parameters** (Ω_m, σ8) that generated a universe, from a single field map — a compact, quantitatively-scored world-model probe.

**Why this is novel.** Self-supervised learning has been applied to CAMELS with **generative** models (VAE / autoencoder) and **contrastive** models (SimCLR-style). This is, to my knowledge, the **first joint-embedding-predictive (JEPA / LeJEPA)** approach on CAMELS — the non-generative, non-contrastive paradigm — brought here at scale with FSDP.

**Shipped so far:**
- Curation of 12 both-suite fields into one pooled SSL corpus (~180k maps), with disk-cached manifests.
- A from-scratch **ViT-L JEPA (~210M params)** trained with **LeJEPA / SIGReg** anti-collapse under **FSDP + bf16** on 2 GPUs — a stable 1000-step run (target-embedding std steady ~0.9, SIGReg at floor; 72 samples/s, 12.2 GB/GPU peak). Effective-rank monitoring then caught what the healthy std masked: *dimensional* collapse (rank ≈ 2), now addressed with a covariance regularizer + target normalization (see the probe plan).
- A frozen-encoder **cosmology probe**: a small `(μ, σ)` moment head regressing Ω_m/σ8, with a latent-space atlas for interpretability.

### Cosmology probe — 🚧 work in progress

**Goal.** Freeze the pretrained JEPA encoder and ask a direct question: *did label-free self-supervised pretraining actually learn cosmology?* Train only a small moment head on the frozen features to predict (Ω_m, σ8), and compare against the supervised CMD `o3_err` CNN benchmark — in-suite (IllustrisTNG) vs held-out suite (SIMBA) to test cross-simulation robustness.

**Achieved so far.** The pipeline runs end to end — raw fields → curated → trained at scale → served → probed. On a **single field (Mgas)** with a **1000-step (under-trained) checkpoint**, the frozen encoder shows a **real but weak** cosmology signal: **Ω_m R² ≈ 0.23** (positive — the encoder genuinely encodes cosmology), σ8 near zero (expected — σ8 is harder from single fields), with roughly-calibrated uncertainty. Crucially, the probe head converges almost immediately, so the ceiling is the *representation*, not the probe — diagnosed as under-training plus a low effective rank from strong anti-collapse regularization.

**Plan to get it working:**
1. **Scale pretraining** — 1000 steps (~64k samples) is a smoke run; real SSL needs 100k+ steps. This is the single biggest lever.
2. **Fix the dimensional collapse at its source.** SIGReg's per-projection (marginal Cramér–Wold) test is nearly blind to *anisotropic* collapse when total variance is spread right — measured gradient ≈ 2e-4 at rank-2 vs ≈ 1.2 for a covariance penalty. Added alongside SIGReg: a VICReg-style **variance-hinge + off-diagonal covariance** term; **target LayerNorm** (removes the shrink-to-constant collapse driver); **exact periodic augmentation** (roll/rot/flip — the CAMELS box is periodic) for view diversity; and **multi-block masking** (target ratio 6% → 22%) to demand richer features. `eff_rank` is now the tracked success metric, gated by a short 300-step sweep before the keeper.
3. **Attentive probe readout** — a learnable-query attention pool replaces mean-pooling (DINOv2 / I-JEPA eval standard); the encoder stays frozen, so it's still an honest test of the features.
4. **Probe all 6 parameters** — expect high R² on Ω_m/σ8 and near-zero on the 4 astrophysical nuisance params (as the supervised benchmark shows), which would demonstrate the **astro-insensitivity** that the SSL-cosmology literature explicitly wants.
5. **Held-out SIMBA** cross-suite robustness eval (normalizing with the training suite's statistics, per the benchmark convention).
6. **Multi-field input channels** — the strongest published SSL result on CAMELS uses multi-channel input.

## Highlights (the parts worth reading)
- **SIGReg is distributed-friendly by construction** — its anti-collapse regularizer is an expectation over the batch, so at scale you just **all-reduce per-GPU partial statistics** for the global-batch statistic; no cross-device negative-pair gathering (contrast: SimCLR). Shipped with a correctness test: world=2 × batch-B ≡ world=1 × batch-2B in loss *and* gradient.
- **bf16 is the throughput lever, activation-checkpointing the memory lever** — measured, not assumed (see Results).
- **Curation is a *policy*, not a constant** — rejection thresholds read off the empirical distribution's tail; disk-cached manifests make reruns instant and turn N re-scans into 1.
- **Collapse is monitored, not hoped for** — target-std *and* effective rank (participation ratio) are logged every run, which surfaced a genuine low-rank finding rather than hiding it.

## Repository layout

```
src/            # production engine — the dataset-agnostic library + CAMELS pipeline
  jepa_loss.py    JEPA model + LeJEPA loss (pure library, no toy-training code)
  sigreg.py       SIGReg + VICReg var/cov regularizers; `--verify` distributed gate
  train_fsdp.py   FSDP/DDP + bf16 distributed trainer
  probe.py        frozen-encoder cosmology probe (attentive head, moment loss, atlas)
  data/           CAMELS field loader + curation
scripts/        # production drivers (run the probe, analyze fields)
study/          # the from-scratch fundamentals — imports the library from src/, nothing here is imported back
  collapse_study.py       synthetic study: stop-grad vs EMA vs SIGReg (what actually stops collapse)
  sigreg_demo.py          SIGReg sanity demo (~0 for N(0,I), large for collapsed)
  analysis/               ablations (SIGReg frequency/dimension blind-spot, failure modes)
  notes/                  study notes + logged results
```

The split is one-directional: `study/` depends on `src/`, never the reverse — so the production engine carries no pedagogical code, and the learning artifacts stay runnable against the real library.

## Quickstart
```bash
pip install -r requirements.txt

# Stage 3 — distributed LeJEPA training on CAMELS fields (2 GPUs)
# SIGReg (--sigreg-lambda 0.7) prevents COMPLETE collapse but is nearly blind to ANISOTROPIC
# (dimensional) collapse -- healthy per-dim std yet eff_rank ~2. The VICReg var/cov terms
# (--var-coef/--cov-coef) and target normalization (--target-norm) supply the strong, correctly-
# directed gradient against low rank that SIGReg lacks. Watch `eff_rank` in the logs.
torchrun --standalone --nproc_per_node=2 src/train_fsdp.py \
  --mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 \
  --var-coef 1e-2 --cov-coef 2e-2 --target-norm \
  --d 1024 --layers 24 --heads 16 \
  --steps 1000 --batch 32 --save /workspace/ckpt.pt

# Verify distributed SIGReg is all-reducible (expect the ×world grad match)
torchrun --standalone --nproc_per_node=2 src/sigreg.py --verify

# Cosmology probe on a frozen checkpoint
python scripts/run_probe.py --ckpt /workspace/ckpt.pt --field Mgas --epochs 20
```

## Results (logged as built)
- [x] **Method (Stage 2):** from-scratch JEPA collapse study — stop-gradient (not EMA decay) is the load-bearing anti-collapse mechanism; symmetric variant collapses (std→0, loss→0) as predicted. Distributed SIGReg all-reduce verified.
- [x] **Distributed training (Stage 3), video ViT-B, 2×A40, 114.5M params:** DDP/fp32 → FSDP+bf16 = **146 → 492 samples/sec (3.4×)**, MFU 7.7% → 25.7%; activation checkpointing cuts peak memory **12.1 → 1.6 GB (7.5×)** at ~20% throughput cost. bf16 engages tensor cores; FSDP-vs-DDP throughput-neutral at this scale. Full table: [`notes/day4_results.md`](notes/day4_results.md).
- [x] **CAMELS Stage-1 training, ViT-L (~210M), 2× RTX 4000 Ada, FSDP+bf16+LeJEPA:** stable 1000-step keeper, target-std steady ~0.9, SIGReg at floor, no collapse; 72 samples/s, 884 ms/step, 12.2 GB/GPU peak, 17% MFU.
- [ ] **Cosmology probe:** 🚧 in progress — real signal (Ω_m R²≈0.23) on an under-trained checkpoint; scaling pretraining + tuning underway (see above).
- [ ] **ViT-L FSDP sweep:** 4-row ddp/fsdp/fsdp+bf16/fsdp+bf16+ckpt at ViT-L scale (where sharding starts to pay vs the FSDP-neutral ViT-B).
- [ ] **Inference optimization (Stage 4):** latency p50/p99 + throughput per lever.

## Roadmap
The engine is dataset-agnostic, so each stage swaps only the loader + input dims:
1. **Stage 1 — CAMELS 2D fields (static, now).** Prove the pipeline end to end on a benchmarked scientific task.
2. **Stage 2 — CAMELS 3D grids across redshift.** The only registered temporal axis in CAMELS (z = 0, 0.5, 1, 1.5, 2) → genuine spatiotemporal prediction of structure formation over cosmic time.
3. **Stage 3 — SDOML solar observations.** NASA Solar Dynamics Observatory ML dataset — real *observed* multi-waveband video at scale; forecast-next-frame = world dynamics.

## Status
Active engineering build. Not affiliated with Meta or AMI Labs. Demonstrates production-engineering skills for self-supervised world-models: large-scale data curation, distributed training with principled anti-collapse, and (next) inference optimization.

— [github.com/KoushikVGitHub/vjepa2-engine](https://github.com/KoushikVGitHub/vjepa2-engine)
