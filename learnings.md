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
