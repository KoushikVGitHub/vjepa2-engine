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
- A from-scratch **ViT-L JEPA (~210M params)** trained with **LeJEPA / SIGReg** anti-collapse under **FSDP + bf16** on 2 GPUs.
- **A diagnosed-and-fixed dimensional collapse.** Effective-rank monitoring caught what a healthy target-std masked: the embedding lived in a ~2-dim subspace of 1024. SIGReg *registered* it but couldn't act — measured gradient ≈ 2e-4 at the collapsed point vs ≈ 1.25 for a covariance penalty (~6000×). Adding VICReg var/cov + target normalization took **effective rank 2 → 38** and the probe **Ω_m R² 0.23 → 0.50**. Full write-up: [`study/notes/collapse_resolution.md`](study/notes/collapse_resolution.md).
- A frozen-encoder **cosmology probe**: an attentive-pool `(μ, σ)` moment head regressing Ω_m/σ8, with a latent-space atlas for interpretability.

### Cosmology probe — 🚧 work in progress

**Goal.** Freeze the pretrained JEPA encoder and ask a direct question: *did label-free self-supervised pretraining actually learn cosmology?* Train only a small moment head on the frozen features to predict (Ω_m, σ8), and compare against the supervised CMD `o3_err` CNN benchmark — in-suite (IllustrisTNG) vs held-out suite (SIMBA) to test cross-simulation robustness.

**Achieved so far.** The pipeline runs end to end — raw fields → curated → trained at scale → served → probed — and label-free pretraining demonstrably learns cosmology. On a **single field (Mgas)** with a **1000-step (still under-trained) checkpoint**:

| | Ω_m | σ8 |
|---|---|---|
| **R²** (frozen encoder, attentive probe) | **0.50** | 0.31 |
| RMSE (physical) | 0.082 | 0.098 |
| Coverage (0.68 ideal) | 0.56 | 0.56 |
| *Supervised CMD `o3_err` CNN, RMSE* | *0.025* | *0.045* |

Ω_m > σ8 is the correct recoverability ordering. We trail the supervised CNN by ~3× — expected for a *frozen, label-free* probe that lacks the supervised model's circular-padding and 8× rot/flip inductive biases; the honest question here is *how close label-free gets*, not whether it wins. Coverage 0.56 means σ is slightly overconfident (the calibration term tightens with training). The probe head's validation loss flattens by epoch 14 while **nothing in pretraining plateaued** — so the ceiling is the *representation*, and encoder steps are the lever.

**Next:**
1. **Is the *pooled* rank the real ceiling?** `eff_rank` is measured over all patch tokens; the probe pools each image to one vector. CAMELS fields are smooth → intra-image tokens are redundant, so pooled rank may be far below the token rank of 38. [`scripts/rank_report.py`](scripts/rank_report.py) measures both on a frozen checkpoint (plus the pooled PCA spectrum and a closed-form ridge on top-k PCs) — this gates whether more steps or a code change is the right spend.
2. **Scale pretraining** — 1000 steps (~64k samples) is a smoke run; real SSL needs 100k+.
3. **Probe all 6 parameters** — expect high R² on Ω_m/σ8 and near-zero on the 4 astrophysical nuisance params (as the supervised benchmark shows), which would demonstrate the **astro-insensitivity** that the SSL-cosmology literature explicitly wants.
4. **Held-out SIMBA** cross-suite robustness eval (normalizing with the training suite's statistics, per the benchmark convention).
5. **Multi-field input channels** — the strongest published SSL result on CAMELS uses multi-channel input.

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
- [ ] **Pooled-rank diagnostic + 10k-step keeper:** does token rank 38 survive per-image pooling, and does scaling steps lift R² further.
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
