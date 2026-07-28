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
| **4 · Inference optimization** | `scripts/bench_infer.py` | Baseline encoder inference → bf16, FlashAttention, `torch.compile`, int8 PTQ, batch sweep. Latency (p50/p99) + throughput + MFU, lever-by-lever. **bf16 = 3.8×, +compile 4.7× total; MFU ceiling ~22%.** | ✅ |

## Current application — Stage 1: CAMELS cosmology fields

[CAMELS](https://camels.readthedocs.io) provides 2D maps of 13 physical fields (gas/dark-matter/stellar density, temperature, HI, velocity, …) from thousands of simulated universes. The task: recover the **cosmological parameters** (Ω_m, σ8) that generated a universe, from a single field map — a compact, quantitatively-scored world-model probe.

**Why this is novel.** Self-supervised learning has been applied to CAMELS with **generative** models (VAE / autoencoder) and **contrastive** models (SimCLR-style). This is, to my knowledge, the **first joint-embedding-predictive (JEPA / LeJEPA)** approach on CAMELS — the non-generative, non-contrastive paradigm — brought here at scale with FSDP.

**Shipped so far:**
- Curation of 12 both-suite fields into one pooled SSL corpus (~180k maps), with disk-cached manifests.
- A from-scratch **ViT-L JEPA (~210M params)** trained with **LeJEPA / SIGReg** anti-collapse under **FSDP + bf16** on 2 GPUs.
- **A diagnosed-and-fixed dimensional collapse.** Effective-rank monitoring caught what a healthy target-std masked: the embedding lived in a ~2-dim subspace of 1024. SIGReg *registered* it but couldn't act — measured gradient ≈ 2e-4 at the collapsed point vs ≈ 1.25 for a covariance penalty (~6000×). Adding VICReg var/cov + target normalization took **effective rank 2 → 38** and the probe **Ω_m R² 0.23 → 0.50**. Full write-up: [`study/notes/collapse_resolution.md`](study/notes/collapse_resolution.md).
- A frozen-encoder **cosmology probe**: an attentive-pool `(μ, σ)` moment head regressing Ω_m/σ8, with a latent-space atlas for interpretability.

### Cosmology probe — 🚧 work in progress

**Goal.** Freeze the pretrained JEPA encoder and ask a direct question: *did label-free self-supervised pretraining actually learn cosmology?* Train only a small moment head on the frozen features to predict (Ω_m, σ8), and compare against the supervised CMD `o3_err` CNN benchmark — in-suite (IllustrisTNG) vs held-out suite (SIMBA) to test cross-simulation robustness.

**Achieved so far.** The pipeline runs end to end — raw fields → curated → trained at scale → served → probed — and label-free pretraining demonstrably learns cosmology. The investigation has moved well past the original 1000-step smoke run (Ω_m R² 0.50): the encoder is now trained to convergence, the tokenizer has been isolated as the Ω_m bottleneck, and the mask-ratio lever has been ruled out. Current best on a single field (**Mgas**, in-suite IllustrisTNG), frozen **patch-8 ViT-L** encoder:

| frozen probe, Mgas in-suite | Ω_m R² | σ8 R² |
|---|---|---|
| **patch-8 ViT-L (converged, current best)** | **~0.60** | **0.388** |
| power-spectrum floor (32-number FFT, no learning) | 0.818 | 0.331 |
| moments (mean/std/skew/kurt, 4-number) | 0.491 | 0.234 |

**σ8 already clears the power-spectrum floor** (0.388 vs 0.331) — the encoder captures non-Gaussian information a 2-point statistic can't see. **Ω_m does not yet** (~0.60 vs 0.818); Ω_m is almost entirely 2-point-saturated, so matching a power spectrum there is the hard, still-open problem. The diagnosis below traced the remaining Ω_m gap to the *linear* patch tokenizer band-limiting high-k power — the hypothesis Phase 2 is now built to test. Nothing here is overstated: Ω_m is **not** solved, and the conv-stem is a hypothesis under test, not a proven win.

**Phase-by-phase status** (the encoder/tokenizer investigation within Stage 1):

| Phase | What | Status |
|---|---|---|
| **0 · Rank / tokenizer disambiguation** | Is the Ω_m plateau rank-limited or architectural? Establish patch-8 vs patch-16. | ✅ **done** — `eff_rank` caps at ≈ 0.55× global-batch and at the ~32-dim intrinsic cosmology dim; clearing the 32-dim floor did **not** move Ω_m ⇒ **tokenizer-limited, not rank-limited**. patch-8 beats patch-16 (Ω_m +45% rel); patch-8 linear tokenizer band-limits high spatial frequency. |
| **1 · Mask-ratio sweep (σ8 lever?)** | Sweep mask ratio via n-blocks 4/8/12 (≈ 25/50/75%) on the patch-8 ViT-L. | ✅ **done — verdict SATURATED.** σ8 flat/non-monotone (0.384 → 0.376 → 0.388, spread 0.011 = noise) ⇒ mask ratio is **not** a σ8 lever; keep n-blocks=4. σ8 stays above the pk-floor at every ratio; Ω_m stays ~0.52–0.60. |
| **2 · track 1 — conv-stem tokenizer (code)** | Add `--stem {linear,conv}`: `log2(patch)` stride-2 3×3 convs with **circular padding** (CAMELS boxes are periodic) + GroupNorm/GELU, then 1×1 conv to the embed dim; same token grid out. | ✅ **done + committed** (`844916c`). Default `linear` is byte-identical (legacy checkpoints load unchanged); CPU test verifies shapes, bit-identity, circular equivariance (max err 8e-7), and a full JEPA step on both stems. |
| **2 · track 2 — train the conv-stem (A/B)** | Train conv vs linear at matched steps/batch/seed on a 24–32 GB 2-GPU pod; report in-suite Ω_m vs the ~0.60 linear baseline and the 0.818 pk-floor. | 🚧 **in progress / next.** Read: conv Ω_m rising toward 0.818 ⇒ tokenizer band-limiting confirmed + fixed; flat near 0.60 ⇒ ceiling is capacity/data, not the tokenizer. |
| **3 · SIMBA cross-suite transfer** | Frozen IllustrisTNG encoder → held-out SIMBA; the headline robustness claim. | 📋 **planned** — success = retention (SIMBA R² / ITNG R²) beats the power spectrum's retention, a win possible at modest absolute R². |

**Still open (beyond the phase ladder):** probe all 6 parameters (expect near-zero R² on the 4 astrophysical nuisance params — the **astro-insensitivity** the SSL-cosmology literature wants), and multi-field input channels (the strongest published SSL result on CAMELS is multi-channel).

## Highlights (the parts worth reading)
- **SIGReg is distributed-friendly by construction** — its anti-collapse regularizer is an expectation over the batch, so at scale you just **all-reduce per-GPU partial statistics** for the global-batch statistic; no cross-device negative-pair gathering (contrast: SimCLR). Shipped with a correctness test: world=2 × batch-B ≡ world=1 × batch-2B in loss *and* gradient.
- **bf16 is the throughput lever, activation-checkpointing the memory lever** — measured, not assumed (see Results).
- **Curation is a *policy*, not a constant** — rejection thresholds read off the empirical distribution's tail; disk-cached manifests make reruns instant and turn N re-scans into 1.
- **Collapse is monitored, not hoped for** — target-std *and* effective rank are logged every run. That caught a *dimensional* collapse hiding behind a perfectly healthy target-std, and fixing it lifted the probe 2.2× ([the debugging story](study/notes/collapse_resolution.md), including why the loss going **up 150×** meant the model got healthier).

## Repository layout

```
src/            # production engine — the dataset-agnostic library + CAMELS pipeline
  jepa_loss.py    JEPA model + LeJEPA loss (pure library, no toy-training code)
  sigreg.py       SIGReg + VICReg var/cov regularizers; `--verify` distributed gate
  train_fsdp.py   FSDP/DDP + bf16 distributed trainer
  probe.py        frozen-encoder cosmology probe (attentive head, moment loss, atlas)
  data/           CAMELS field loader + curation
scripts/        # production drivers
  run_probe.py    train + evaluate the cosmology probe on a frozen checkpoint
  rank_report.py  representation-geometry report: token vs pooled effective rank, PCA, ridge probe
  analyze_all.py  per-field statistics -> curation thresholds (see study/notes/camels_field_stats.md)
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
# directed gradient against low rank that SIGReg lacks.
#
# The coefficients below are TUNED, not defaults: var must DOMINATE cov (~125:1 here; canonical
# VICReg is ~25:1). var is the SCALE knob, cov is the RANK knob -- with cov >= var the model
# minimizes covariance by shrinking every embedding to the origin instead of decorrelating
# (measured: cov -> 0.006 "perfect" while eff_rank hit the floor at 1.0). See
# study/notes/collapse_resolution.md.
#
# Watch eff_rank / tgt_std / var / pred -- NOT the total loss, which is ~92% the var/cov terms
# (the collapsed run scored loss 0.008; the healthy one scores ~1.2).
torchrun --standalone --nproc_per_node=2 src/train_fsdp.py \
  --mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 \
  --var-coef 5.0 --cov-coef 4e-2 --target-norm \
  --d 1024 --layers 24 --heads 16 \
  --steps 1000 --batch 32 --save /workspace/ckpt.pt

# Verify distributed SIGReg is all-reducible (expect the ×world grad match)
torchrun --standalone --nproc_per_node=2 src/sigreg.py --verify

# Cosmology probe on a frozen checkpoint
python scripts/run_probe.py --ckpt /workspace/ckpt.pt --field Mgas --epochs 20

# Representation-geometry report on a frozen checkpoint (no training):
# token eff_rank vs POOLED eff_rank (what the probe actually consumes), the pooled PCA
# spectrum, and a closed-form ridge probe on the top-k PCs -- i.e. how many dims carry
# cosmology rather than nuisance variance.
python scripts/rank_report.py --ckpt /workspace/ckpt.pt --field Mgas --n 3000
```

## Results (logged as built)
- [x] **Method (Stage 2):** from-scratch JEPA collapse study — stop-gradient (not EMA decay) is the load-bearing anti-collapse mechanism; symmetric variant collapses (std→0, loss→0) as predicted. Distributed SIGReg all-reduce verified.
- [x] **Distributed training (Stage 3), video ViT-B, 2×A40, 114.5M params:** DDP/fp32 → FSDP+bf16 = **146 → 492 samples/sec (3.4×)**, MFU 7.7% → 25.7%; activation checkpointing cuts peak memory **12.1 → 1.6 GB (7.5×)** at ~20% throughput cost. bf16 engages tensor cores; FSDP-vs-DDP throughput-neutral at this scale. Full table: [`study/notes/day4_results.md`](study/notes/day4_results.md).
- [x] **CAMELS Stage-1 training, ViT-L (~210M), FSDP+bf16+LeJEPA:** stable 1000-step keeper; 72 samples/s, 884 ms/step, 12.2 GB/GPU peak, 17% MFU (2× RTX 4000 Ada).
- [x] **Dimensional collapse diagnosed and fixed:** effective rank **2 → 38.7**, target-std → 0.96. Root cause = SIGReg's marginal test is near-blind to anisotropic collapse (‖∇‖ ≈ 2e-4 vs 1.25 for a covariance penalty); fix = VICReg var/cov at a var-dominant ~125:1 ratio + target LayerNorm. Three failed gates isolated **var = the scale knob, cov = the rank knob**. [`study/notes/collapse_resolution.md`](study/notes/collapse_resolution.md).
- [x] **Cosmology probe (in-suite, Mgas, 1000-step ckpt):** **Ω_m R² = 0.50**, σ8 = 0.31 — a **2.2× lift** from the rank fix. Label-free pretraining learns cosmology; ~3× off the supervised CNN.
- [x] **Rank vs tokenizer disambiguated (Phase 0):** `eff_rank` caps at ≈ 0.55× global-batch; clearing the ~32-dim intrinsic floor (rank ~25 → ~35+) left Ω_m flat at ~0.63 ⇒ **tokenizer-limited, not rank-limited**. patch-8 beats patch-16 (Ω_m +45% rel; σ8 0.37 > pk-floor 0.331).
- [x] **Mask-ratio sweep (Phase 1): SATURATED** — σ8 flat across ~25/50/75% masking (0.384 / 0.376 / 0.388) ⇒ mask ratio is not a σ8 lever; keep n-blocks=4.
- [ ] **Conv-stem A/B (Phase 2 track 2):** does an overlapping, circular-padded conv tokenizer push Ω_m past ~0.60 toward the 0.818 pk-floor. Code committed (`844916c`); training next.
- [ ] **ViT-L FSDP sweep:** 4-row ddp/fsdp/fsdp+bf16/fsdp+bf16+ckpt at ViT-L scale (where sharding starts to pay vs the FSDP-neutral ViT-B).
- [x] **Inference optimization (Stage 4):** latency p50/p99 + throughput + MFU per lever (`study/notes/day5_inference.md`).

## Roadmap
The engine is dataset-agnostic, so each stage swaps only the loader + input dims:
1. **Stage 1 — CAMELS 2D fields (static, now).** Prove the pipeline end to end on a benchmarked scientific task.
2. **Stage 2 — CAMELS 3D grids across redshift.** The only registered temporal axis in CAMELS (z = 0, 0.5, 1, 1.5, 2) → genuine spatiotemporal prediction of structure formation over cosmic time.
3. **Stage 3 — SDOML solar observations.** NASA Solar Dynamics Observatory ML dataset — real *observed* multi-waveband video at scale; forecast-next-frame = world dynamics.

## Status
Active engineering build. Not affiliated with Meta or AMI Labs. Demonstrates production-engineering skills for self-supervised world-models: large-scale data curation, distributed training with principled anti-collapse, and inference optimization.

— [github.com/KoushikVGitHub/vjepa2-engine](https://github.com/KoushikVGitHub/vjepa2-engine)
