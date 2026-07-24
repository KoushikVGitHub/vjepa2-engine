# Learnings — VISReg × Cosmology Collapse Experiment

A running, linear record of what this project taught me, plus the skills it exercises.
Domain: self-supervised (JEPA) pretraining of a ViT-L encoder on CAMELS 2D cosmology field
maps, probed for (Ω_m, σ8) parameter inference. Hardware: 2× GPU, FSDP + bf16.

---

## The problem in one line

A masked-prediction JEPA trained on **smooth, low-intrinsic-dimensionality** scientific fields
**dimensionally collapses** — it keeps unit per-dimension variance but traps all information in
~8–12 directions — and standard distributional regularizers don't catch it.

---

## Linear learnings (in the order we found them)

**L1 — Distributional priors are blind to anisotropic collapse.**
SIGReg / VISReg enforce that each 1-D marginal looks Gaussian, but their gradient nearly
*vanishes* in the dimensional-collapse basin (‖∇‖ ≈ 2e-4 for the marginal test vs ≈ 1.25 for a
covariance term). So `tgt_std` reads healthy (~0.99) while `eff_rank` sits at ~8. **A healthy
per-dimension variance does not mean a healthy representation.**

**L2 — The covariance term is the load-bearing fix, and we know *why*.**
Effective rank obeys `r_eff = D / (1 + cov_loss)` — the off-diagonal covariance penalty sits
directly in the rank denominator. Minimising it *mechanically* buys rank. Winning recipe:
`--var-coef 5.0` (scale anchor) `--cov-coef 4e-2` (rank knob) `--target-norm`.

**L3 — It's not the data, and it's not the paradigm.**
Pure VISReg collapses on *natural images* too (STL-10, rank ~11), and even in VISReg's **own
native multi-crop paradigm** (rank ~8–9). So collapse under pure distributional regularization
is systemic to this style of SSL — not a smooth-field quirk and not a masked-prediction quirk.
That promotes the covariance term from "a fix for cosmology" to "the load-bearing fix, full stop."

**L4 — The controlled A/B: decorrelation escapes collapse and ~2× the downstream R².**

| | pure VISReg | + covariance |
|---|---|---|
| eff_rank | **11.7** (collapsed) | **72** (escaped) |
| R² Ω_m | **0.235** | **0.493** |
| R² σ8 | 0.276 | 0.311 |
| RMSE Ω_m / σ8 | 0.102 / 0.100 | 0.083 / 0.098 |

**L5 — The honest ceiling (Track-1 baseline) reframes "useful".**
A *32-number radial power spectrum* — pure numpy, no learning — infers Ω_m at **R² 0.818**,
near the supervised ceiling. Our best SSL encoder (0.493) is **below** that floor.

| classical feature | dim | R² Ω_m | R² σ8 |
|---|---|---|---|
| power spectrum P(k) | 32 | **0.818** | 0.331 |
| moments (mean/std/skew/kurt) | 4 | 0.491 | 0.234 |
| P(k) + moments | 36 | 0.823 | **0.463** |

So we are **not** at the Gaussian plateau — we're *below* it. The masked pretext leaks 2-point
information a trivial FFT captures for free. Ω_m is almost entirely 2-point (moments add ~nothing);
**σ8 is where non-Gaussian information lives** (moments lift it 0.33 → 0.46).

**L6 — Rank is a *diagnostic*, not the objective.**
Small effective rank is not intrinsically wrong — representation learning *is* compression. Low
rank is only pathological when it coincides with *low R²* (information lost), vs optimal when it's
a genuine sufficient statistic (information kept). Rank 72 is already *above* the ~32-dim intrinsic
task dimension (ridge R² saturates at k≈32), so the extra dims are nuisance. **The right objective
is: minimise rank subject to holding R² — optimise information, not rank.**

**L7 — Ideal R² is a ladder, not 1.0.**
R² = 1 is impossible (cosmic variance: a single map is one random realisation). The real ceiling is
the supervised-CNN / power-spectrum bar (~0.82 on Ω_m for a feedback field like Mgas). σ8 sits
systematically lower at every rung and is the harder, more interesting target.

**L8 — Engineering / infra learnings.**
- **Throughput:** batch 32 → 128 lifted MFU ~7% → **18.6%** (bigger matmuls, better PCIe
  compute/comms ratio); peak memory only 5.6 GB of 24 GB — the card was 95% idle at batch 32.
- **Batch helps the statistic too:** more tokens per step → a better-conditioned covariance estimate.
- **RunPod NFS root-squash** breaks defaults: `~/.cache/torch/kernels` isn't writable (JIT kernels
  recompile every launch) and `tar` can't chown — fixed via `PYTORCH_KERNEL_CACHE_PATH` and
  `--no-same-owner`.
- **Frozen-feature caching** makes the probe ~15× faster (encoder is frozen → embed once, reuse).
- **Total loss is not a health metric** — a collapsed run can have *lower* loss than a healthy one;
  watch `eff_rank` and the singular-value spectrum, not the loss.

**L9 — Where this points next (Track 3, now necessary not optional).**
To justify SSL here the encoder must (a) *reach* the power spectrum's 0.82 on Ω_m — harder masking
(higher ratio / multi-block) so the pretext can't shortcut the 2-point structure; and (b) *exceed*
pk on σ8 — a non-Gaussian-sensitive target (wavelet/scattering coefficients, or the residual after
removing the radial power spectrum). Concrete gap to close: **+0.33 on Ω_m** just to match classical.

---

## Skills exercised (mapped to the world-model / AMI goal)

**Deep (the vertical of the "T") — SSL & training dynamics**
- Self-supervised representation learning: JEPA / masked prediction, VISReg, SIGReg, VICReg,
  multi-crop invariance — and implementing a loss (VISReg) faithfully from the paper.
- Diagnosing **representation collapse**: effective rank (participation ratio), covariance
  structure, singular-value spectra, distinguishing complete vs anisotropic/dimensional collapse.
- Loss-function design and reasoning about *why* a term works (the `r_eff = D/(1+cov_loss)` identity),
  not just that it works.

**Broad (the horizontal) — systems, science, method**
- **Distributed training:** PyTorch FSDP / DDP, `torchrun`, multi-GPU sharding, bf16 mixed
  precision, activation checkpointing, MFU measurement and throughput tuning.
- **Scientific ML:** cosmology domain (CAMELS multifield maps, Ω_m / σ8 inference), power spectra,
  moments, non-Gaussian statistics, intrinsic-dimensionality reasoning.
- **Experimental design:** controlled A/B with one variable changed, classical baselines as
  information ceilings, ablations, separating *correlation* (rank↑) from *causation* (R²↑).
- **Data engineering:** memory-mapped datasets, offline curation + manifest caching, frozen-feature
  caching, multi-field corpus pooling.
- **MLOps / infra:** RunPod GPU pods, SSH automation (key management, detached `setsid nohup` runs,
  log-polling watchers that survive disconnects), GPU scheduling, environment debugging.
- **Numerical / statistical methods:** ridge regression (closed-form, dependency-free), FFT-based
  radial power spectra, R² / RMSE evaluation, sim-level train/val splits to prevent leakage.
- **Software engineering:** clean segregation of concerns (a `LOSS_MODES` capability registry),
  reproducible git-driven workflow, portfolio-quality documentation.
- **Scientific judgement:** steelmanning an external model's (wrong) claim against measured evidence,
  and updating my own framing (rank-as-objective → rank-as-diagnostic) when the data demanded it.

**Why it matters for the goal:** this is exactly the AMI-Labs shape of work — improving on EMA /
stop-gradient heuristics with distributional + decorrelation priors, on a real world-model encoder,
with the evidence discipline to know when a representation is genuinely learning the physics versus
gaming the pretext.
