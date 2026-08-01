# Learnings — VISReg × Cosmology Collapse Experiment

**This file is my (Claude's) continuous-learning knowledge base for this project** — the goals we set,
the decisions and *why*, the dead-ends, and the kill-criteria — so each session compounds on the last
instead of re-deriving. Domain: self-supervised (JEPA) pretraining of a ViT-L encoder on CAMELS 2D
cosmology field maps, probed for (Ω_m, σ8) inference. Hardware: 2× RTX 4090, FSDP + bf16.

---

## Goals (the north stars)

- **Immediate (✅ answered):** does adding covariance decorrelation to VISReg turn a *collapsed*
  cosmology encoder (rank ~8, R² 0.23) into a *useful* one, measured by probe R²? → **Yes** — rank
  11.7 → 72, Ω_m R² 0.235 → 0.493.
- **Project claim (Track 3, in progress):** one frozen JEPA encoder that learns cosmological
  parameters from smooth, low-intrinsic-dim fields and **transfers across simulation suites**
  (IllustrisTNG → SIMBA) with **better retention than a power spectrum** — evidence of learned
  physics, not curve-fitting.
- **Meta:** keep this file compounding — record goal, decision, *why*, and kill-criteria every session.

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

**L10 — Infra learnings from the A4000 pod (2026-07-25, running the patch-8 A/B).**
- **2-GPU FSDP deadlocks on A4000s (no NVLink).** First-collective NCCL hang; tell-tale =
  **100% GPU util but only ~55% of TDP power** (76 W / 140 W) and *step-0 never logs*. Real matmuls
  pull near TDP (~135 W). Fix here: **run single-GPU per arm, both GPUs in parallel** (`ab_1gpu.sh`) —
  world_size=1 makes NCCL a no-op, and concurrency recovers the wall-time. (Untried alt:
  `NCCL_P2P_DISABLE=1`.) *Lesson: at "100% util, no progress", check power draw before assuming compute.*
- **`pkill -f <name>` kills its own SSH shell.** `pkill -f train_fsdp` matches the `bash -c` running it
  (its argv contains the string) → SIGKILLs the parent → command returns with **no output and nothing
  launched**. Use the self-excluding regex trick: `pkill -9 -f "train_[f]sdp"`, or `tmux kill-server`.
- **patch-8 ViT-L is memory-bound on 16 GB.** 1024 tokens × L24, and the **SIGReg loss scales with
  token count** (`sigreg.py` projections) → OOM in backward even at batch 32 (~14 GB). Matched **batch 16**
  is the safe fit (~11.7 GB). Keep both arms at the same batch or the A/B is confounded.
- **Detach with tmux, never `nohup … &` over one-shot SSH.** The backgrounded job races the channel-close
  SIGHUP and dies (no log written). `tmux new-session -d` fully detaches and survives drops.

**L11 — Tokenizer A/B result (2026-07-25): patch-8 HELPS (Ω_m +50% rel), but the batch mattered decisively.**
patch-16 vs patch-8, cov-recipe, 1000 steps, single-GPU-parallel on 2× A4000, Mgas in-suite.
**Ran it TWICE — the batch size flipped the conclusion:**

| arm | batch 16: Ω_m / σ8 | **eff-batch 32: Ω_m / σ8** |
|---|---|---|
| patch-16 (control) | 0.126 / 0.211 | 0.197 / 0.225 |
| patch-8 (test) | 0.130 / **0.061** | **0.296 / 0.240** |
| Δ (p8 − p16) | +0.004 / **−0.150** | **+0.099 / +0.015** |
| pk floor | 0.818 / 0.331 | 0.818 / 0.331 |

**At batch 16, patch-8 looked flat/worse — a FALSE NEGATIVE.** At effective batch 32 (grad-accum,
micro-16×2), patch-8 is **clearly better: Ω_m 0.197→0.296 (+50% rel)**, σ8 slightly up. **⇒ the
linear-patch-embed band-limiting hypothesis IS supported** — finer patches recover high-k signal, and the
gain concentrates on **Ω_m** (the 2-point/high-k-dominated parameter, exactly where patch-16 averaging
loses most). **Lesson: patch-8's 4× token space needs adequate effective batch to pay off; undertrained,
its bigger representation *hurts*.** Two caveats: (1) it **narrows but does not close** the gap — 0.296 «
pk's 0.818, both arms still undertrained at 1000 steps (batch-64 keeper alone = 0.49); (2) settle the true
converged gap with the Track-3 convergence run (both to plateau).

Infra to make batch 32 fit on 16 GB: patch-8 true batch-32 OOMs by ~300 MB (SIGReg loss ∝ 1024-token
count) → added **`--grad-accum`** (effective batch = batch×accum at one-micro-batch peak mem; batch-stat
losses stay per-micro-batch). Both arms matched at micro-16×accum-2 to keep the A/B clean.

**L13 — Convergence result (2026-07-25, global-batch 64, 4000 steps): patch-8 wins decisively AND beats pk on σ8.**
Seeded probe ladder, Mgas in-suite, 2-GPU global batch 64 (rank ~25):

| step | patch-16 (Ω_m / σ8) | patch-8 (Ω_m / σ8) |
|---|---|---|
| 2000 | 0.350 / 0.284 | 0.626 / 0.363 |
| 4000 | 0.433 / 0.295 | **0.630 / 0.367** |
| pk floor | 0.818 / 0.331 | 0.818 / 0.331 |

- **Tokenizer band-limiting CONFIRMED, large:** patch-8 Ω_m 0.63 vs patch-16 0.43 (+0.20, +45% rel), σ8 0.37 vs 0.30.
- **patch-8 σ8 0.367 > pk-alone 0.331** — the encoder beats a 2-point statistic on the *non-Gaussian* parameter
  (pk+moments 0.463 still leads σ8, but beating pk-alone is the signal the transfer story needs).
- **Ω_m narrowed not closed:** 0.63 « pk 0.818 (Ω_m is 2-point-saturated — pk hard to beat there).
- **Convergence:** patch-8 **plateaued** (0.626→0.630 by step 2000); patch-16 **still climbing** (0.35→0.43).
- ⚠ **plateau is likely RANK-limited, not fundamental:** rank ~25 < 32-dim floor = the **16 GB A4000 ceiling**
  (global-64 is the max batch patch-8 fits). 24 GB cards (batch 64/GPU → rank ~72) could push patch-8 past 0.63.
- **Decision:** patch-8 is the encoder going forward. Exit gate: σ8 ✓ (0.367 ≥ 0.33), Ω_m marginal (0.630 < 0.65,
  plausibly hardware-capped). The batch-16 A/B verdict (L11) was DOUBLY undertrained (batch + steps); this is the real one.

**L12 — eff_rank is capped by the BATCH (samples in the covariance estimate), not just steps.**
The feature covariance is estimated over N samples per step, so `eff_rank ≤ N`. Empirically rank ≈
0.55–0.59 × global-batch at 1000 steps (still *climbing*): **rank 38 @ batch 64, rank 72 @ batch 128.**
So global batch 32 caps rank ~12 — **below the ~32-dim intrinsic cosmology dimension** (ridge R²
saturates at k≈32) → **R² is bottlenecked for a batch reason, masquerading as a convergence plateau.**
⚠ **grad-accum does NOT fix this** — the batch-statistic is computed per micro-batch, so accum-2 keeps a
16-sample statistic and rank stays ~12. Only a genuinely larger batch (or 2-GPU **distributed** SIGReg,
which all-reduces the statistic over the global batch) raises the sample count. **Fix: run global batch
64** (2-GPU × 32/GPU, distributed SIGReg = 64-sample stat) → rank ~38 ≥ 32, the keeper's 0.50 regime.
Going to 128 (rank 72) doesn't improve R² — cosmology only needs ~32 dims, so 64 is the efficient sweet
spot. *Takeaway: when a JEPA probe R² plateaus low, check `eff_rank` vs both the intrinsic task dim AND
the batch ceiling before blaming training length.*

**L14 — Architecture sequence on a 24 GB card (2026-07-26): resolve the L12↔L13 rank tension, then conv-stem.**
Open contradiction in this log: **L12** inferred (from the ~32-dim intrinsic dim) that global batch 64 → rank ~38 ≥ 32 is
enough and "128 won't help"; but **L13** (the real patch-8 convergence run) *observed only rank ~25 at global 64* — below
the 32-dim floor — with Ω_m capped at 0.63. Both can't hold. Note the ceiling tracks **global batch** (samples in the
all-reduced SIGReg statistic), not per-card VRAM: a *single* 24 GB fits only ~batch 48–56 for patch-8 (unsharded params) →
global ≤ ~56, **no gain over today's global 64**. Need **2× 24 GB @ batch 64/GPU → global 128 → rank ~72** for the test to bite.
Sequence (each phase runs at the Phase-0 global batch so rank is never a hidden variable):
- **Phase 0 — rank control / Ω_m disambiguation.** patch-8 ViT-L, n-blocks 4, global 128. Ω_m > ~0.70 ⇒ L13 right, plateau was
  rank-limited (L12's "64 enough" was over-optimistic for patch-8's 4× tokens). Ω_m still ~0.63 at rank 72 ⇒ L12 right, rank
  is NOT the bottleneck ⇒ tokenizer band-limiting ⇒ conv-stem warranted. This run is also the rank-matched CONTROL downstream.
- **Phase 1 — mask sweep (σ8 lever).** `mask_sweep.sh` at BATCH=64, n-blocks 4/8/12. Push σ8 above the 0.367 that already beats pk.
- **Phase 2 — conv-stem + circular-padding ViT (runs regardless).** Two arms (hygiene): (2a) plain ViT patch-8 + **circular
  padding** (physically exact for CAMELS periodic boxes, near-free), (2b) **conv-stem tokenizer** + circular padding. Control =
  Phase-0/1 best plain-ViT patch-8 at matched global batch. Q: does the conv-stem inductive bias push Ω_m toward 0.818 *beyond*
  what rank alone bought? Needs code (conv-stem patch-embed + circular padding). Phase 0 must precede it so the conv-stem's
  Ω_m payoff isn't re-confounded with batch/rank.

**L15 — Phase 0 execution on 2× RTX 4090 24 GB (2026-07-26, VERDICT: TOKENIZER-LIMITED).**
- **Pod/infra:** fresh image has **tmux NOT preinstalled** (`apt-get install -y tmux`). Same *persisted* `/workspace`
  volume survives (all ckpts + `data/` corpus intact). The `runpod_auto` pubkey must be **re-added to
  `~/.ssh/authorized_keys` on every new pod** (the passphrased `id_ed25519` can't be used non-interactively).
- **VRAM reality (corrects L13's guess):** patch-8 **batch 64/GPU OOMs a 24 GB card** (tried to alloc ~4 GB over) —
  the L13 "batch 64/GPU fits 24 GB → rank ~72" estimate was **WRONG**. **batch 48/GPU fits** at ~23.7 GB/GPU
  (~0.8 GB headroom) ⇒ **global 96** (not 128) ⇒ rank ~37–53, still clears the 32-dim floor with margin.
- **2-GPU health tell on 4090 (no NVLink):** `NCCL_P2P_DISABLE=1` required; healthy = 100% util **at ~410 W/GPU**
  (near the 450 W TDP). The NCCL hang looks identical on util (100%) but draws only ~55% TDP — **power draw, not
  util, is the diagnostic.**
- **Early rank climb (global 96):** eff_rank **1.8 → 13.6 by step 100** (cov still 20.9 → still rising), already
  tracking to beat the global-64 run's ~25 plateau. Rate ~2.3 s/step → 4000 steps ≈ 2.5 h + ~15 min probe.
- **Probe gotchas (patch-8 is slow):** each frozen-feature probe ≈ 25–35 min (1024 tokens × 15k maps × 3 splits
  through the 202M ViT-L, then 20-epoch head fit). Run the two ladder probes on separate GPUs in parallel.
- **The step-4000 ckpt was TRUNCATED (738 MB vs the valid 844 MB)** — the disk-quota-exceeded that killed the
  in-line probe phase also corrupted the *final* save. Fix: freed ~16 GB (deleted 21 stale ckpts) and used the
  valid **step-3000** ckpt as the converged endpoint (eff_rank had already plateaued ~35–38 by step 2600, so 3000
  is scientifically the "converged" read). Lesson: **check `/workspace` quota headroom BEFORE a run that saves a
  ladder** — 4 × 844 MB ckpts + a 53 GB data corpus is tight under the 75 GB per-user quota.
- **VERDICT — TOKENIZER-LIMITED (Ω_m), rank cleared:** probe ladder (Mgas in-suite R², seed 0):
  global-64 ref **Ω_m 0.630 / σ8 0.367** → global-96 @2000 (rank ~35) **0.584 / 0.375** → @3000 (rank ~35–38)
  **0.630 / 0.390**. Ω_m returned to **exactly 0.630** *after* rank climbed 25 → 35+ (clearing the 32-dim floor
  with margin) — the rank-limited hypothesis needed Ω_m > 0.70, so **rank is NOT the bottleneck.** σ8 improved
  0.367 → 0.390 (extra rank helps the non-Gaussian channel, still beats the pk-floor 0.331) but Ω_m is flat: the
  classic **tokenizer band-limiting** signature — patch-8's *linear* patch-embed averages away the high-k power
  Ω_m rides on, and no amount of rank recovers it. **⇒ resolves L12↔L13 in favour of neither: the plateau is
  architectural. Next = build the conv-stem + circular-padding ViT (Phase 2), NOT more batch/rank.** Reusable
  rank-matched control ckpt: `ckpt_p8_b128_step3000.pt`.

**L16 — Phase 1 (mask sweep) COMPLETE 2026-07-27 — VERDICT: SATURATED, mask ratio is NOT a σ8 lever.**
- All three mask arms trained and saved to `/workspace/checkpoints/` (844 MB each, seed 0, patch-8 recipe, global
  batch 96 = batch 48/GPU, STEPS=2000): `ckpt_m4.pt` (~25% mask, eff_rank 35.1 = Phase-0 control ✓), `ckpt_m8.pt`
  (~50%), `ckpt_m12.pt` (~75%, eff_rank ~34.5 — rank parity holds across all three). m4/m8 trained on 2× RTX 4090;
  m12 on 2× RTX PRO 4500. `/workspace` network volume persists across pod stop AND across *different* pods
  (verified 3×: re-attached same checkpoints on new IPs / new GPU types).
- **RESULT — σ8-vs-mask curve** (Mgas in-suite R² on IllustrisTNG, patch-8 ViT-L, seed 0), probed on 2× RTX A4000:

  | n-blocks (~mask %) | Ω_m R² | σ8 R² |
  |---|---|---|
  | 4  (~25%) | 0.577 | 0.384 |
  | 8  (~50%) | 0.516 | 0.376 |
  | 12 (~75%) | 0.604 | 0.388 |
  | pk-alone floor | 0.818 | 0.331 |

- **VERDICT — SATURATED.** σ8 = 0.384 → 0.376 → 0.388 is **non-monotone and flat** (whole spread 0.011 = noise);
  no rise with harder masking. Per the pre-registered rule (monotone rise above 0.390 ⇒ lever; flat/declining ⇒
  saturated ⇒ keep n-blocks=4), **mask *ratio* is not the non-Gaussian lever → keep n-blocks=4 for the conv-stem
  (Phase 2).** σ8 stays ~0.38 (> pk-floor 0.331) regardless of ratio: the non-Gaussian signal comes from the
  patch-8 tokenizer, not from masking amount. Ω_m stays far below the 0.818 pk-floor at every ratio — reconfirms
  Phase-0 tokenizer-limit. NOTE: Track-3 step 2 (L249+) distinguishes mask *ratio* (tested here, dead) from mask
  *geometry* (contiguous `8×1` vs scattered `4×4`, same ratio) — geometry is the untested lever; revisit only if
  the conv-stem's σ8 also refuses to move.
- **⚠ PROBE-SPEED LESSON (corrects the old "run probes sequentially" note).** The earlier "probes are DataLoader-
  bound, ~30–50 min, parallel halves throughput" diagnosis was WRONG about the cause: `run_probe.py` defaults to
  **`--workers 4`**, which starves the frozen-feature precompute (GPU sits at 0% util / 16 W while it waits on 4
  loader procs). On a 128-core pod, `--workers 32` alone takes the GPU to 100%; and because the box has cores to
  spare, running two probes **in parallel on the two GPUs** (m4→GPU0, m8→GPU1) is a genuine ~2× win, not a wash.
  Fixed sweep script `scripts/phase1_probes.sh` (scp'd to pod): WAVE1 = m4+m8 parallel @32 workers, WAVE2 = m12
  @48. Whole 3-arm sweep ran in ~45 min vs the old ~2–2.5 h estimate. Probe cmd unchanged (features are
  deterministic — shuffle off, augment off — so worker count changes speed only, never the numbers).
- **⚠ BLACKWELL (sm_120) POD SETUP — the stock RunPod image FAILS on Blackwell and must be patched first:**
  the image ships `torch 2.4.1+cu124` whose archs stop at sm_90 → any CUDA kernel dies with "no kernel image for
  sm_120". Fix (driver 580 already supports it): `pip install --upgrade --index-url
  https://download.pytorch.org/whl/cu128 torch torchvision` (gets torch 2.11.0+cu128 / torchvision 0.26.0+cu128,
  which include sm_120). This upgrade lives in the CONTAINER, not `/workspace` — **redo it on every fresh Blackwell
  pod.** Also torch≥2.7's `torchrun` added `--duplicate-std*-filters`, making bare `--d` an ambiguous abbreviation
  it steals → **use `--edim` not `--d`** (alias added to `train_fsdp.py`, commit 9c32a46; `phase1_resume.sh`
  already uses it). On an Ada pod (4090/L4, sm_89) none of this is needed — stock image just works.
- **GPU shopping note (confirmed):** for this workload the binding constraint is 32-dim rank floor → need global
  batch ≥~96 = batch 48/GPU × 2. RTX PRO 4500 (32 GB, ~200 W) fits batch 48 with ~9 GB headroom, ~1.9–2.6 s/step
  (a bit slower than 4090's 2.0). L4 (22.5 GB) can't fit batch 48 → would force rank <35 → **avoid.**

- **Reminder:** before the SIMBA cross-suite download (Phase 3), bump `/workspace` storage to ~150 GB (currently
  60 G of the 75 GB quota — SIMBA is another ~53 GB and will not fit).

**L17 — Phase 2 (conv-stem tokenizer) STAGED 2026-07-27 — code ready, no pod used yet.**
- **Thesis:** Ω_m is capped at ~0.60 vs pk-floor 0.818 because the **linear** patch-8 embed (`nn.Linear(patch*patch, d)`
  on *disjoint* patches) is a box-average that band-limits high-k. Fix = an overlapping conv tokenizer that spans
  patch boundaries. σ8 is already past the 2-pt floor (0.388 > 0.331) so this phase targets **Ω_m only**.
- **Flag: `--stem {linear,conv}`** (default `linear`). New `ConvStem` in `src/jepa_loss.py` = `log2(patch)` stride-2
  3×3 convs (patch-8 → 3 layers, channels 1→128→256→512) + GroupNorm + GELU, then a 1×1 conv → d, with
  **`padding_mode="circular"`** because CAMELS maps are periodic boxes (zero/reflect padding would inject a fake
  edge). Output is the SAME (B, grid², d) token grid in the same row-major order → pos-embed / predictor / probe all
  unchanged. GroupNorm not BatchNorm (BN running-stats are an FSDP+bf16 hazard, same reason as the VISReg projector).
- **Default-off is bit-identical:** when `stem="linear"` only `self.proj` is built (conv branch never constructed),
  RNG draw order is unchanged, and the state_dict carries exactly the legacy keys (`proj/pos/blocks`) → **existing
  m4/m8/m12 checkpoints still load.** Conv checkpoints carry disjoint `conv_stem.*` keys (no collision).
- **Wired through:** `src/train_fsdp.py` (`--stem` arg → `build_model` → `ViTEncoder`), `scripts/run_probe.py`
  (`--stem` arg → `enc_cfg` → `load_frozen_encoder`, which passes `**enc_kw` straight through — no edit needed in
  `src/probe.py`). Probe `--stem` MUST match how the ckpt was trained.
- **Verified on CPU** (`scripts/test_conv_stem.py`, all green): token shapes match, linear path bit-identical +
  legacy-keys-only, conv keys disjoint, both stems deterministic, **circular equivariance holds** (periodic shift of
  the input rolls the token grid by one column, max err 8e-07), and both stems run masked-forward + a full JEPA step.
  (Test uses real stem geometry — 1024 tokens, d=1024 — but shallow layers: torch's **CPU** build segfaults on the
  true 24-layer attention; transformer depth is irrelevant to tokenizer shape-compat.)
- **Phase-2 launch = `scripts/phase2_convstem.sh`** (controlled A/B: trains BOTH stems at identical
  steps/batch/seed=0, n-blocks=4 fixed, then probes each). **Needs a 24–32 GB 2-GPU pod** (batch 48/GPU → global 96
  for rank parity; A4000 16 GB can't *train* this). Launch:
  `tmux new-session -d -s p2 "bash /workspace/scripts/phase2_convstem.sh > /workspace/logs/p2.log 2>&1"`.
  Read: conv Ω_m rising toward 0.818 ⇒ tokenizer band-limiting confirmed+fixed (adopt conv for Phase 3);
  flat near 0.60 ⇒ ceiling is capacity/data, not the tokenizer.

**L18 — Phase 2 (conv-stem tokenizer) COMPLETE 2026-07-28 — VERDICT: conv-stem is a DECISIVE Ω_m lever. Band-limiting CONFIRMED and FIXED. Adopt conv-stem for Phase 3.**
- **Result (in-suite Mgas R², patch-8 ViT-L, global-batch 64, seed 0):**

  | stem   | Ω_m R² | Ω_m RMSE | σ8 R² |
  |--------|--------|----------|-------|
  | linear | 0.5456 | 0.0787   | 0.3724 |
  | conv   | **0.7661** | **0.0565** | **0.4196** |
  | pk-floor | 0.818 | — | 0.331 |

- **Ω_m: +0.22 absolute (0.546→0.766, +40% rel), −28% RMSE.** Conv closes ~80% of the linear→pk gap (gap 0.272 → 0.052). The overlapping stride-2 circular convs recover the high-k power the disjoint linear patch-embed box-averaged away. **The Phase-0 diagnosis (tokenizer-limited, not rank-limited) is now causally confirmed** — swapping ONLY the tokenizer moved Ω_m by 0.22.
- **σ8 also rose to 0.420** (from 0.372), a new high, further above the pk-floor 0.331 — conv improves non-Gaussian extraction too, not just Ω_m.
- **The rank caveat resolved in the GOOD direction.** Both arms ran at global-batch 64 (20 GB RTX 4000 Ada cap; batch 32/GPU) → eff_rank plateaued ~21, BELOW the ~35 the batch-96 baseline reached (batch-64 cost the linear arm ~0.06: 0.546 here vs 0.604 at batch 96). But conv beats linear by +0.22 at *matched* rank → the win is unambiguously the TOKENIZER, not a rank artifact. Corollary: **0.766 is a conservative lower bound** — conv at batch 96 (rank ~35) should read higher. Worth a confirmation run on a 24 GB pod, but the verdict doesn't depend on it.
- **Infra:** pod = 2× RTX 4000 Ada 20 GB @157.157.221.29:24595 (stock torch 2.4.1+cu124, sm_89 → no Blackwell patching). Conv-stem peaks **18.0 GB/GPU @batch 32** (+1.3 GB over linear's 16.7); batch 40 would be tight on 20 GB, batch 48 needs 24 GB+. tmux `p2`, `BATCH=32 PEAK=153`. Both probes ran in parallel one-per-GPU (`--workers 32`, GPUs 100% during precompute — Phase-1 starvation avoided). Both arms `train rc=0`, both probes `rc=0`. Runtime ~7 h wall (2×4000 steps ~3 s/step + parallel probes ~15 min).
- **⇒ Phase 3 (SIMBA cross-suite transfer) uses the conv-stem encoder** (`ckpt_stem_conv.pt`). The headline metric stays the ITNG→SIMBA retention ratio (SSL vs pk); conv now gives Phase 3 a tokenizer that actually carries high-k Ω_m info into the transfer test. Before the SIMBA download, bump `/workspace` to ~150 GB (still 60 G of 75 GB quota).

---

**L19 — Phase 2b/2c (batch-96 confirmation + reviewer-control battery) 2026-07-29 — conv arm LANDED (antigravity, 32 GB pod); linear/mlp/H3/H5 + S1 done. ⚠ ONE DECISIVE DISCREPANCY OPEN (linear 0.638 vs 0.767) that controls the whole verdict — see below. Early signal REVISES L18 against interest: capacity/nonlinearity — not overlap — explains much of the tokenizer gain, the corrected pk floor is HIGHER, and the transformer LOSES in-suite information.**
- **Why this phase.** L18's "decisive lever / band-limiting confirmed" rested on a *single* 2-arm A/B at eff_rank ~21, one seed, no control arms, and against a **mis-computed** pk floor. An independent audit (`assumptions.md`) + a reviewer pass raised H1/H3/H4/H5/H7/H9 and S1/S2/S3. Phase 2b/2c runs the matched battery at batch 96 to pin the *cause*.
- **S1 (pk baseline fairness) — DONE & MEASURED.** The old pk floor (0.818 / 0.331) was the *model-selection (val) R²* on a **different** split. Rebuilt `ps_baseline.py` to score the probe's exact `sim_split(seed=0)` test sims (α on val, report **test**; guarded by `test_ps_baseline_split.py`). **Fair pk floor (Mgas test): Ω_m 0.834, σ8 0.446; pk+moments 0.837/0.544** — *higher* than the old numbers. ⇒ **L18 claims RETRACTED:** conv σ8 0.420 does **not** clear the fair floor (0.446); "conv closes ~80% of the pk gap" was computed against the wrong floor. In-suite, SSL does not beat pk on **either** parameter.
- **linear / mlp — DONE (3 head-seeds, batch 96 / rank ~35, in-suite Mgas encoder):** linear **0.638 ± 0.012** / σ8 0.401 ± 0.006 ; mlp **0.723 ± 0.002** / σ8 0.375 ± 0.001. Linear rose 0.546 @b64 → **0.638 @b96**, so *part of L18's +0.22 was rank*, not tokenizer.
- **H1 (capacity vs overlap) — PRELIMINARY, decisive-looking.** `mlp` is a **disjoint**, param-matched (~2 M), nonlinear per-patch tokenizer with **no cross-patch overlap** — yet it beats linear by **+0.085 Ω_m** with tiny error bars. ⇒ capacity + nonlinearity is a real Ω_m lever *without* overlap. **The L18 "overlap / band-limiting" story is likely wrong**; the disjoint-conv control (S3 `convdisjoint`) is the decider and is still training.
- **H3 (raw tokenizer stage) — SURPRISE (most consequential):** probing the **pre-transformer** tokens gives linear-tok **0.878**, mlp-tok **0.904** — *higher* than the full pretrained encoder (0.638 / 0.723) **and above the fair pk floor**. **The masked-JEPA transformer is *degrading* the linearly-decodable in-suite Ω_m signal.** Legitimate (frozen encoder, head trained on train sims, tested on held-out sims). ⇒ the in-suite SSL representation is *worse* than its own raw patch embeddings; **cross-suite transfer (Phase 3) is now the *only* viable headline**, not in-suite accuracy.
- **H5 (Mcdm generality) — DONE (1 seed):** linear 0.606 / 0.653 ; mlp 0.675 / 0.748. mlp > linear holds on a 2nd field; **Mcdm σ8 (0.65–0.75) ≫ Mgas σ8 (0.40)** — cold dark matter carries σ8.
- **conv @b96 — LANDED (antigravity, 32 GB RTX PRO 4500 Blackwell, seed 0):** Ω_m **0.7968** / σ8 **0.4158** (eff_rank 34.9, loss 1.275, peak 24.61 GB). So 0.766 @b64/rank21 did survive at rank ~35 — conv reads *higher* at batch 96, as L18 predicted.
- **⚠ OPEN DISCREPANCY — DECISIVE, must resolve before any verdict.** Antigravity's matched A/B compares conv against a **linear = 0.767 / 0.415 at seed 0**. Our established linear @b96 is **0.638 ± 0.012 / 0.401 (3 head-seeds, seed 1234)**. Same nominal arm, **0.13 Ω_m apart = 10× the error bar.** This flips the entire story two ways:
  - If linear ≈ 0.638 → conv gain **+0.16** (large; band-limiting partly revives).
  - If linear ≈ 0.767 → conv gain **only +0.030**, AND mlp 0.723 falls *below* linear ⇒ **H1 inverts** (linear ≥ mlp ≥ conv within noise).
  - **Most likely cause:** antigravity's 0.767 is the **same-seed (0)** partner to conv; our 0.638 is **seed 1234**. If so, linear swings **0.13 across pretraining seed — larger than every arm-to-arm delta (conv +0.030, mlp +0.085).** That would mean the tokenizer deltas are **not significant against pretraining-seed noise**, and **S2 (seed spread) becomes the headline control, not a footnote.** Unconfirmed — needs the raw `/workspace` logs (no SSH to the 32 GB pod at time of writing). **Do NOT quote a conv-vs-linear gain until the two linear runs are reconciled at matched seed.**
- **STILL PENDING (on `/workspace`, uncommitted — pull tomorrow via `scripts/phase2_verdict.py --logs /workspace/logs`):** **convdisjoint (S3 overlap decider)**, **conv-tokenizer (H3, does conv's gain sit pre-transformer?)**, **H7 random-init floor**, **H9 strict-hygiene**, **S2 pretraining-seed spread (now the decisive control — see discrepancy above)**, conv-Mcdm (H5). `phase2_verdict.py` auto-resolves the read rules once these land.
- **Process / infra lessons.** (1) conv OOMs at batch 48 on a 24 GB card (~25.5 GB; linear/mlp fit); split the run — linear/mlp on the 24 GB 4090, conv+ on a **32 GB pod at matched batch 48**, coordinating through the shared RunPod **network volume** (`mfs…runpod.net`, survives stopping either pod) with a second (antigravity) agent — a filesystem *blackboard*, not A2A. (2) **torchrun ≥ 2.7 rejects `--d`** as an ambiguous abbreviation of `--duplicate-stdout-filters`; renamed the trainer flag **`--d` → `--edim`** (accepts both), version-independent fix. (3) **Aggregator bug caught by "test & prove":** the IN-SUITE block prints **RMSE / R2 / Coverage**; a new scraper matched the first `Omega_m=` (RMSE 0.07) not R2 (0.638) — caught because the printed aggregate contradicted the raw log; fixed by anchoring to `R2 :`. (4) A3: `[0-9.]+ → -?[0-9.]+` so the H7 negative floor isn't silently dropped.
- **Net so far (provisional):** the headline is shifting from "conv-stem band-limiting win" to a more honest, more interesting story — **in-suite, SSL is below its own raw tokenizer and below pk; the value proposition rests entirely on cross-suite *transfer*.** Conv landed at 0.797 but the conv-vs-linear *gain* is **undefined until the 0.638/0.767 linear conflict is settled** — that reconciliation, not the conv number, is the gating result.
- **NEXT SESSION (tomorrow) — ordered checklist.** (1) Get the 32 GB pod SSH (or have antigravity commit `/workspace/logs`); run `python scripts/phase2_verdict.py --logs /workspace/logs`. (2) **Reconcile linear: which seed is antigravity's 0.767?** Read its probe log header — if it's seed 0, we have a same-seed A/B (conv +0.030) *and* a 0.13 seed swing on linear ⇒ **S2 is the story.** If it's seed 1234, it contradicts our 0.638 3-seed mean → a probe/config bug to hunt. (3) Pull convdisjoint (S3), conv-tok (H3), H7, H9, S2 → fill the verdict table. (4) Only then write the final L19 verdict + close CONTEXT_GRAPH decision row 16/18. (5) If deltas survive seed noise → Phase 3 (SIMBA cross-suite: bump `/workspace` to ~150 GB, T5 norm injection). If not → the honest paper is "in-suite tokenizer deltas are within seed noise; transfer is the only signal."


**L20 — Phase 2c in-suite bundle CLOSES L19's open verdict (2026-08-01) — S2 kills the seed-swing scare; S3 revives overlap; in-suite is settled, transfer is the headline.**
- **Why this entry.** L19 landed the conv arm (0.797 @b96) but left the verdict *undefined*, gated on ONE discrepancy: is linear 0.638 (our seed 1234) or 0.767 (antigravity seed 0)? If 0.767, every tokenizer delta drowns in pretraining-seed noise. Phase 2c ran the remaining controls — S2 (2nd pretraining seed per arm), S3 (`convdisjoint` overlap decider), H9 (strict hygiene) — on an A40, then probed all 5 arms × 3 head-seeds on a cheap RTX-4000-Ada (frozen encoder, in-RAM cache), A100 released as soon as training finished.
- **In-suite Mgas / IllustrisTNG, R² (mean ± std over head-seeds 0–2), batch-96, rank ~35:**

  | Arm | Stem | Ω_m | σ8 | seeds |
  |---|---|---|---|---|
  | conv_s0 | overlapping conv | **0.809 ± 0.007** | 0.389 ± 0.013 | pretrain s0 (today) |
  | conv_reprobe (p2b_conv) | overlapping conv | 0.795 ± 0.002 | 0.413 ± 0.014 | p2b run |
  | conv_h9 | overlapping conv, held-out test sims | 0.787 ± 0.004 | 0.420 ± 0.007 | hygiene |
  | convdisjoint | conv kernel=stride (NO overlap) | 0.748 ± 0.008 | 0.427 ± 0.011 | S3 decider |
  | mlp (L19) | disjoint MLP tokenizer | 0.723 ± 0.002 | 0.375 ± 0.001 | H1 capacity |
  | linear_s0 | linear patch-embed | 0.651 ± 0.016 | 0.355 ± 0.006 | **pretrain s0 (today)** |
  | linear (L19) | linear patch-embed | 0.638 ± 0.012 | 0.401 ± 0.006 | pretrain s1234 |
  | — fair pk floor (L19-S1) — | — | 0.834 | 0.446 | — |
  | — raw tokenizer (L19-H3) — | linear-tok 0.878 / mlp-tok 0.904 | — | — |

- **S2 — RESOLVES L19's decisive discrepancy AGAINST the seed-swing hypothesis.** New linear pretrain-seed-0 = **0.651 ± 0.016** sits next to our seed-1234 **0.638 ± 0.012**. Linear's pretraining-seed spread is **~0.013, not 0.13.** Conv likewise: p2b 0.795 vs seed-0 0.809 → spread ~0.014. Every arm's seed spread (~0.013) is ≈10× smaller than the conv-vs-linear gap (~0.15). ⇒ **Antigravity's linear 0.767 is an outlier (config/probe difference), not a pretraining-seed effect** — flag for a config check, but it is NOT load-bearing. **The feared "tokenizer deltas within seed noise" scenario does NOT hold; the deltas are significant.**
- **S3 — REVIVES the overlap lever L19-H1 tentatively dismissed.** L19 saw mlp beat linear without overlap and guessed "overlap story likely wrong" — but mlp only isolates *capacity*. The `convdisjoint` decider (matched conv capacity, kernel=stride, NO cross-patch overlap) now lands at 0.748, giving a clean matched-batch decomposition (linear mean ~0.645):
  - capacity / nonlinearity (linear→mlp): **+0.078**
  - conv inductive bias, still disjoint (mlp→convdisjoint): **+0.025**
  - **overlap (convdisjoint→conv): +0.061** — ~5× the ~0.01 run noise.
  ⇒ **overlap IS a real, separable ~0.06 Ω_m lever on top of capacity.** Both capacity and overlap contribute; L19's "overlap likely wrong" is reversed by its own named decider.
- **H9 — hygiene clean.** conv_h9 0.787 ≈ conv_reprobe 0.795 (within 0.008 across strictly held-out test sims). The conv score is not a leakage artifact.
- **conv-vs-linear gain is now DEFINED (L19 said undefined):** conv ~0.80 vs linear ~0.645 = **+0.15 Ω_m, real and seed-robust.** No conv σ8 win (conv 0.39–0.42 ≈ linear 0.40).
- **In-suite parity remains a dead end (L18/L19 stand, reinforced):** best conv 0.809 < fair pk 0.834 on Ω_m and 0.389 < 0.446 on σ8; and per L19-H3 the raw tokenizer (0.878 / 0.904) beats BOTH the transformer and pk ⇒ masked-JEPA *degrades* in-suite decodable signal. **In-suite accuracy is not the value proposition.**
- **VERDICT.** The in-suite battery is settled: **conv (capacity + overlap) is the best in-suite arm, +0.15 Ω_m over linear, seed-robust, hygiene-clean, overlap causally isolated (S3).** It still loses in-suite to pk and to its own raw tokenizer. Therefore the only remaining headline is **cross-suite TRANSFER**: freeze conv_s0, probe on SIMBA, show ITNG-trained-norm retention beats pk's cross-suite drop. **Green-light Phase 3.** conv_s0 = the encoder.
- **Method note (already fixed upstream).** Probe logs print RMSE / R2 / Coverage — a `tail -1` collation grabs Coverage, not R2; anchor to `R2 :`. This was already fixed on origin/main (9f98d60 + phase2_verdict.py, bfd938b); re-derived here when a monitor's collation showed ~0.60 instead of ~0.80.
- **Infra.** A100-80GB train-only (batch-96 SIGReg needs the rank); frozen-encoder probing offloaded to a cheap RTX-4000-Ada mounting the SAME eur-is-1 network volume (dual-mount confirmed) — expensive card released the instant training finished. SIMBA Mgas maps (Phase 3) pulled by a 2-vCPU CPU pod over plain HTTPS from `users.flatironinstitute.org/~camels/CMD/2D_maps/data/SIMBA/` (Globus not needed; path verified by matching the known ITNG file's byte size).

---

## Track 3 — the plan (settled via design review, 2026-07-24)

Decided through a structured design interrogation ("grill-me"), ordered by dependency.

**Value proposition (the claim).** *Not* "beat the power spectrum on Ω_m" — a losing fight, since
Ω_m is 2-point-saturated (moments add ~nothing: 0.818 → 0.823). The claim is **cross-suite transfer
/ relative robustness**: one frozen encoder, pretrained on IllustrisTNG, that **retains more of its
accuracy on held-out SIMBA than a power spectrum does** — evidence it learned generalizable physics,
not per-suite curve-fitting. Multifield synergy is an opportunistic free-rider, tested only if it
doesn't cost the transfer path.

**Sequencing — in-suite first, then transfer.** A sub-classical in-suite representation makes a weak
transfer headline; you can't claim "it learned physics that transfers" before it demonstrably learned
the physics well in-suite. So:

1. **Convergence curve — kill the undertraining confound *first*.** The 0.49 was measured at 1000
   steps and never plateaued. Train the winning recipe (`--var-coef 5.0 --cov-coef 4e-2 --target-norm`)
   to ~10k steps, `--save-every 2000`, probe at 2k/4k/6k/8k/10k → the *true* converged baseline plus an
   R²-vs-steps curve. Every later experiment must beat this, not the undertrained number.
2. **Harder masking = geometry, not ratio.** The current mask (4×4 × n_blocks 4, ~25% *scattered*) is
   trivially interpolable on a smooth field — the exact shortcut. Sweep toward **large contiguous**
   target blocks; **`8×1` is the key control** (same 25% ratio as 4×4, only geometry differs → isolates
   shape from amount). Keep the cov term on (a harder task can re-trigger collapse). **Watch σ8** at
   every geometry — for a large hole in a *non-Gaussian* field, the gap between Gaussian interpolation
   and the truth *is* the non-Gaussian signal, so masking is implicitly a σ8 lever too.
3. **Non-Gaussian target — deferred.** Build it (wavelet/scattering, or de-power-spectrum'd residual)
   *only if* σ8 refuses to move after the masking sweep. Keeps lever count minimal and makes the
   non-Gaussian claim earned, not assumed.
4. **Then cross-suite transfer.** Machinery already exists (`run_probe.py`: frozen ITNG encoder +
   ITNG-trained probe → eval SIMBA). Blocked only on SIMBA maps (Globus transfer). **Headline metric =
   ITNG-normalization applied to SIMBA inputs** (true zero-shot — the honest test), with SIMBA-norm
   reported as a decomposition (input-scale vs feature-mismatch). Needs a small `FieldMapDataset` change
   to inject external mean/std. **Success = retention (SIMBA R² / ITNG R²) beats the power spectrum's
   retention** — a win is possible at modest absolute R².

**Exit gate — in-suite phase → transfer (settled Q8).** Stop improving in-suite when **(plateau)** the
convergence curve + masking sweep stop yielding gains **AND (credibility floor)** σ8 ≥ the pk floor
(0.33) *and* Ω_m ≥ ~0.65. Crucially, a **plateau *below* the floor is a publishable kill result** — the
architecture's honest ceiling on smooth-field cosmology — **not** a licence to transfer-test a weak base.

**Statistical rigor (tiered, settled Q9).** Bootstrap CIs on *every* probe R² (resample test sims —
free); screen geometries at 1 seed / plateau step; spend pretraining seeds (2–3, full convergence)
**only on the decisive `8×1`-vs-`4×4` comparison**. Decision rule: believe an improvement only if it
exceeds the bootstrap CI **and** replicates across ≥2 seeds on the decisive config. Every number in
this file carries a ±, not a bare point estimate.

**Tactics (T1–T6, confirmed 2026-07-24).**
- **T1 — Field:** probe **Mgas** primary (feedback differs most across suites → strongest transfer
  story, where a power spectrum is most brittle) + **Mtot** as a clean-field, high-ceiling sanity anchor.
  One multifield encoder, so probing extra fields is cheap.
- **T2 — SIMBA in parallel:** kick off the Globus SIMBA transfer (Mgas, Mtot, `params_LH_SIMBA.txt`)
  *concurrent* with in-suite GPU work — it's I/O, keep it off the critical path.
- **T3 — Pod / data:** `/workspace` is a RunPod network FS → checkpoints + data + Globus persist across
  restart; re-append `runpod_auto.pub` if the SSH endpoint changes; first action on restart = verify
  `ckpt_cov.pt` + the 12 field files are intact.
- **T4 — Ordering:** sequential runs, both GPUs (FSDP) each — curve first (it *sets the plateau step*),
  then geometry screen at that step, then the seeded decisive comparison; probes parallelize one-per-GPU.
- **T5 — Norm change:** implement `FieldMapDataset` external mean/std injection when we reach the
  transfer step, not before.
- **T6 — Cost:** the curve's plateau step caps every later run's length (flatten at 4k ⇒ nothing runs to
  10k) — the main lever against burning GPU-hours.

**Guardrails:** cov term stays on throughout; pod is currently down (restart is the first execution step).

*Design tree complete (design review 2026-07-24) — nothing left to decide; execute on pod restart.*

---

## Encoder architecture — decision (2026-07-24)

**Decision: keep a ViT-JEPA substrate, but move off plain ViT-L toward a small, conv-stem,
periodic-padded ViT — and prove the tokenizer hypothesis *first* with a clean patch-8 A/B.**

**Why not a CNN (asked directly).** "CNN already works" refers to *supervised* CAMELS CNNs — a
different object. This project is **self-supervised** (masked prediction, no labels) + **transfer**;
that thesis is a statement about SSL representations, which a supervised CNN can't demonstrate. Masked
JEPA also wants a ViT (it masks *token subsets* and attends over the visible ones; masked convolution
is awkward). And the artifact's reason to exist is a JEPA/world-model demonstration (AMI target). So we
keep the ViT paradigm — but **import the CNN's inductive bias** (conv stem, translation-equivariance,
periodic padding) rather than ignoring it.

**Why plain ViT-L is likely mismatched.**
- **Over-parameterized:** ~300M params for a task of intrinsic dim ~32 / 2 target scalars → capacity
  buys *nuisance dimensions*, not R² (the very collapse-to-junk we fought). ViT-S/B is better matched.
- **Tokenizer band-limiting (the prime suspect for why SSL < pk):** patch-16 linear embed *averages
  away* sub-patch (high-k) power — exactly where cosmology signal concentrates and exactly what the
  power spectrum sees for free. This is the most plausible mechanism for sitting *below* the pk floor.
- **Wrong priors for the field:** periodic BCs (sim-box slices) + statistical isotropy → circular
  padding + conv equivariance are free correct priors a plain ViT must learn from data.

**The deciding test (staged, cheap, in-budget): patch-16 → patch-8, everything else held.**
Isolate the tokenizer variable — same ViT-L backbone, same cov recipe, and **hold physical mask
geometry fixed** (`--block 8 --n-blocks 4` at patch-8 == the keeper's 64 px blocks at 25 %, vs
`--block 4 --n-blocks 4` at patch-16). 1000 steps first, directly comparable to the 0.493 keeper.
Command staged in `scripts/run_patch8.sh`.
- **R² jumps toward 0.818** ⇒ the linear patch embed was discarding the high-k signal; fix = smaller
  patch / conv stem. Clean mechanistic result *and* a remedy.
- **R² barely moves** ⇒ the bottleneck is the SSL objective, not the tokenizer; architecture isn't the
  lever. Also decisive.

**Variable hygiene — do NOT bundle backbone-downscale with the patch test.** patch-8 (tokenizer) and
ViT-S/B (capacity) are two variables; mixing them confounds. Order: (1) patch-8 tokenizer A/B on ViT-L,
(2) *then* backbone-downscale as a separate arm, (3) conv-stem + circular padding only if patch-8
confirms the tokenizer is load-bearing (that step needs a model-code change, not just flags).

**Sequencing vs Track 3.** Architecture is **upstream of** the convergence curve — changing the encoder
resets every convergence/transfer number. So settle patch-size *before* the long Track-3 runs, or run
patch-8 as a parallel arm of step 1. Caveat: patch-8 = 4× tokens (32×32 grid) → heavier; start at
`--batch 64 --ckpt`, and for the *final* clean number match effective batch to the keeper's 128.

*Note — parked contingency (from the OpenEvolve assessment):* if σ8 stalls through the masking sweep,
OpenEvolve (LLM evolutionary code search, cheap numpy evaluator) is a viable way to *evolve a
non-Gaussian summary statistic* that hardens the classical σ8 baseline — seed = `scripts/ps_baseline.py`.
Not on the critical path; it strengthens the benchmark our robustness claim is measured against.

---

## My toolset for this project (Claude's skills, honed here)

A living operating manual — the capabilities I have access to and the *refined pattern* for using each
on **this** project, so the workflow compounds across sessions instead of restarting cold.

- **Remote execution — SSH bridge to the RunPod pod.** Dedicated passphrase-less key
  (`~/.ssh/runpod_auto`); write run scripts to `/workspace`, launch training **detached** with
  `setsid nohup` so it survives SSH drops, drive both GPUs. Replaces the old "user pastes logs by hand"
  loop. *Gotcha learned:* RunPod NFS root-squashes `$HOME` → set `PYTORCH_KERNEL_CACHE_PATH`, use
  `tar --no-same-owner`.
- **Long-run orchestration — background watchers.** `run_in_background` SSH pollers that block on the
  pod until `=== RESULT ===` or a crash marker, then notify me — no busy-waiting. Parallel probes
  pinned per-GPU via `CUDA_VISIBLE_DEVICES`. *Gotcha:* a 30-min watcher SSH can drop (`Connection reset`)
  — the detached job survives, just reconnect and re-tail.
- **Faithful external research — WebFetch / WebSearch + GitHub MCP.** Verify claims against the *actual*
  source instead of recalling: confirmed VISReg's `num_projections=4096`, that Galaxy10 is a
  *downstream-only* eval (never pretraining), and the ideal-R² ceilings. Rule: read the repo/paper, don't guess.
- **Codebase tools — Grep / Glob / Read / Edit / Write.** Ground every design question in what the code
  *actually does* before recommending — e.g. caught the silent SIMBA-normalization choice in
  `run_probe.py` and the real mask geometry (`block 4 × n_blocks 4`) in `jepa_loss.py`.
- **Version control — git to `main`.** Push so the pod pulls; **no AI-attribution trailers** (your preference).
- **Artifacts — the `Artifact` tool.** Publish this file as a private, shareable page; same URL on every update.
- **Persistent memory — the `memory/` system.** Auto-loads next session, so verdicts, the reframe, and
  the plan survive context resets. `learnings.md` is the repo-side, human-readable companion.
- **Structured design review — the `grill-me` skill.** Dependency-ordered interrogation of a plan; this
  session produced the entire Track-3 design (value prop → sequencing → masking → exit gate).
- **Subagents — the `Agent` tool.** Parallel code-review / research on demand (a review agent previously
  caught 3 bugs, incl. a post-abort save that would have destroyed a checkpoint).

**How to keep this sharp:** whenever a tool saves a cycle — or costs one — note the refined pattern (and
the gotcha) here. This section is meant to *improve* as the project runs: a compounding operating manual,
not a static list.
