# Independent Audit — `vjepa2-engine` / `vjepa2-probe`

**Auditor:** independent read-through of the repository at `27b3bda` (branch `main`, clean tree), 2026-07-29.
**Scope:** all of `src/`, `scripts/`, `tests/`, `.github/`, `README.md`, `learnings.md`, plus a live run of the
CPU test suite. No GPU, no CAMELS data, no checkpoints were available, so every claim that depends on
running the real pipeline is marked **UNVERIFIED** rather than accepted or rejected.

**Method.** Read the code before the prose, then checked the prose against the code. Every finding below
cites `file:line`. Findings are separated into *defects* (something is wrong or can silently go wrong) and
*assumptions* (something the work rests on that is nowhere enforced or verified). The distinction matters:
most of the risk in this project is in the second category.

**Verified locally:** `python -m pytest -q` → **42 tests, 41 passed, 1 skipped** (the gloo world=2 test skips
on Windows, as documented).

---

## 1. Summary

| # | Finding | Severity | Where |
|---|---|---|---|
| A1 | Sim-level split assumes an *identity* curation manifest; nothing enforces it, and the H9 path can mis-align silently | **High** | `probe.py:181`, `run_probe.py:134`, `train_fsdp.py:471` |
| A2 | Every result rests on a single fixed test split (`seed=0`, 100 sims); reported error bars only cover probe-head init | **High** | `probe.py:181`, `phase2b_controls.sh:26` |
| A3 | The result-scraping regex cannot parse a negative R², and can silently substitute the held-out suite's number | **High** | `phase2b_controls.sh:65`, `phase2c_audit.sh:59`, `probe_multiseed.sh:34` |
| A4 | Phase scripts advertise `seed=0` for training but never pass `--seed`; all pretraining actually ran at `1234` | **High** | `phase2b_controls.sh:24`, `mask_sweep.sh:30`, `phase0_rank.sh:20`, `phase2_convstem.sh:34`, `ab_converge.sh:25` |
| A5 | H9 hygiene arm trains on ~10% less data than its comparator — leakage and data volume are confounded | **Medium** | `phase2b_h9.sh:26`, `train_fsdp.py:471` |
| A6 | Probe uses no model selection (val loss computed then discarded); the pk baseline *does* select on val | **Medium** | `probe.py:274`, `ps_baseline.py:92` |
| A7 | Three different R² definitions across the three probes — `rank_report.py` normalizes by the *train* mean | **Medium** | `rank_report.py:123` vs `probe.py:327` vs `ps_baseline.py:99` |
| A8 | Throughput and MFU ignore `--grad-accum`; reported samples/sec understated by that factor | **Medium** | `train_fsdp.py:639`, `train_fsdp.py:356` |
| A9 | Dataset standardization statistics are computed over train+val+test maps, including in the "strict hygiene" arm | **Medium** | `fields.py:208`, `train_fsdp.py:488` |
| A10 | `rank_report.py` is hard-wired to patch-16/linear — it cannot load any checkpoint from Phase 0 onward, yet is in the README quickstart | **Medium** | `rank_report.py:36`, `README.md:147` |
| A11 | `convdisjoint` and `mlp` stems have no CI gate; the S1/H9 index guards are standalone scripts pytest never collects | **Medium** | `tests/test_conv_stem.py:89`, `pytest.ini:2` |
| A12 | Stale / mutually inconsistent numbers between README, `learnings.md` and commit messages | **Medium** | see §5 |
| A13 | `torch.load(..., weights_only=False)` on checkpoint paths taken from the CLI | **Low** | `probe.py:62` |
| A14 | Dead video-curation module with a module-level `decord` import and unbounded recursion; `requirements.txt` pulls a heavy stack nothing current uses | **Low** | `curation.py:10`, `curation.py:59`, `requirements.txt` |
| A15 | One mask is shared by every sample in a batch | **Low** | `jepa_loss.py:91` |
| A16 | Assorted doc drift: "41 tests", gradient "≡", `analyze_all` transform table vs the trainer's | **Low** | see §5 |

Nothing here disputes the *direction* of the headline results. The conv-stem effect (+0.22 Ω_m) is far
larger than any bias I can attribute to these issues. What the issues do affect is the **size of the error
bar** and the **precision of the claims** — and several of them are the kind that fail silently, which is
exactly the failure mode this repo's own CI section says it is defending against.

---

## 2. Correctness findings

### A1 — The sim-level split assumes a curation manifest that nothing checks *(High)*

The correctness note at `probe.py:11-15` is right that a map-level split leaks the label, and `sim_split`
(`probe.py:181`) correctly splits at sim granularity. But the mapping from *dataset position* to *sim* is
`position // 15`, and that is only true when the curation manifest is the identity.

The probe builds its dataset with `min_std=0.05` (`run_probe.py:35`), and `FieldMapDataset` drops any map
failing that threshold (`fields.py:210`). After a drop, dataset position `i` maps to `manifest[i]`, and the
true sim is `manifest[i] // 15` (`fields.py:282`) — not `i // 15`. `sim_split` is then partitioning the
wrong thing, and maps from one simulation can land in both train and test. That is the exact leak the
docstring warns about, reintroduced one layer down.

The only guard is the divisibility assert at `probe.py:199`. It catches most drops but not all: dropping
any multiple of 15 maps passes it. There is no assertion anywhere that `len(ds) == len(ds.maps)`, and
`run_probe.py` never inspects the manifest.

The same assumption propagates:
- `ps_baseline.py:137` indexes the raw `.npy` with **no** curation, so if the probe's dataset ever drops a
  map, the "apples-to-apples" claim at `ps_baseline.py:142-146` quietly becomes false — the two splits
  would cover different sims. `scripts/test_ps_baseline_split.py` proves the two `sim_split` *calls* agree,
  which is a different claim.
- `train_fsdp.py:476` computes `per_field, rem = divmod(len(ds), nf)` for the H9 holdout. If different
  fields drop different numbers of maps but the total happens to stay divisible by 12, `per_field` is
  wrong and the excluded index blocks land on the wrong sims in every field — a silently broken hygiene
  control that still prints a confident `[data] H9 holdout: excluded ...` line (`train_fsdp.py:486`).

`scripts/test_holdout_h9.py` hardcodes `PER = 15000`, i.e. it verifies the arithmetic *given* the
assumption rather than the assumption itself.

**Fix:** assert `len(ds.manifest) == len(ds.maps)` in `run_probe.py` and in the H9 block, or derive the
split from `manifest[i] // maps_per_sim` instead of from positions. One line each, and it converts a silent
leak into a loud failure.

### A2 — A single fixed test split underlies every number in the repo *(High)*

`sim_split` defaults to `seed=0` (`probe.py:181`) and no caller ever overrides it — `run_probe.py:134`
calls `sim_split(len(ds))` bare, `ps_baseline.py:147` passes `seed=args.seed` whose default is 0,
`rank_report.py:205` passes none, `train_fsdp.py:478` hardcodes `seed=0`. Every reported R², every phase
verdict, and the pk floor are measured on the **same 100 test simulations**.

The statistical consequence is understated in the write-ups. The test set contains 1500 maps but only
**100 independent label draws** — the 15 maps of a sim share one `(Ω_m, σ8)`. So the sampling error on any
R² is governed by n=100, not n=1500. The error bars the repo does report (`probe_multiseed.sh:2`,
`phase2b_controls.sh:13`) vary only the **probe head init** (±0.03), and `phase2c_audit.sh:8` adds a second
**pretraining** seed. Neither varies the split. A ±0.03 head-seed band therefore cannot be read as a
confidence interval on the underlying quantity, and comparisons like "0.384 → 0.376 → 0.388, spread 0.011
= noise" (`README.md:54`) are calibrated against the wrong noise source.

This matters most for the small effects: the Phase-1 mask-ratio "SATURATED" verdict and the σ8 movements
(0.367 → 0.390 → 0.420) are all inside a plausible split-to-split band that has never been measured. The
Phase-2 Ω_m effect (+0.22) is almost certainly larger than it.

**Fix:** add `--split-seed` to `run_probe.py`/`ps_baseline.py` and run the two decisive arms over 3 splits.
This is cheap — the probe head is minutes once features are cached — and it is the single highest-value
missing control, above any of the H*/S* hypotheses currently queued.

Statistical nit while here: the aggregators report `st.pstdev` (population σ) over n=3
(`phase2b_controls.sh:67`, `phase2c_audit.sh:61`, `probe_multiseed.sh:36`). With n=3 that biases the spread
downward and is not a standard error. Use sample stdev, and label it as a spread, not an interval.

### A3 — The result scraper drops negative R², and can report the wrong suite *(High)*

All three aggregators use:

```python
re.search(r"R2\s*:\s*Omega_m=([0-9.]+)\s+sigma8=([0-9.]+)", t)
```

(`phase2b_controls.sh:65`, `phase2c_audit.sh:59`, `probe_multiseed.sh:34`). `[0-9.]+` cannot match a leading
minus. `probe.py:329` computes `r2 = 1 - ss_res/ss_tot`, which is negative whenever the head predicts worse
than the test mean.

Two failure modes, both silent:

1. **The H7 control is the arm most likely to produce a negative R².** `--random-init` (`run_probe.py:94`)
   deliberately probes an untrained encoder — the "learned nothing" floor. If that floor lands at, say,
   −0.02, the seed is skipped by the `if m:` guard (`phase2b_controls.sh:66`) and the mean is silently
   computed over fewer seeds, or the label prints `<no result>`. The control designed to establish the
   floor is the one the harness cannot read.
2. **Worse: `re.search` scans the whole log.** `run_probe.py:159-173` prints the in-suite block first and
   the held-out SIMBA block second. If the in-suite R² is negative and the held-out one is not, the regex
   skips past the in-suite section and matches the **SIMBA** line, which is then reported under an
   in-suite label. Note `probe_multiseed.sh:24` gets this right for its per-seed echo (`grep -A2
   "IN-SUITE"`) and then gets it wrong in its own aggregate ten lines later.

**Fix:** `(-?[0-9.]+)`, and anchor the search to the `=== IN-SUITE` block. Better still, have
`run_probe.py` emit one machine-readable line (or a JSON sidecar) and delete the three copies of this
regex.

### A4 — The phase scripts never pass `--seed` to training *(High)*

Every orchestrator defines `SEED=${SEED:-0}` and prints it in its banner — `mask_sweep.sh:30,35`,
`phase0_rank.sh:20,24`, `phase2_convstem.sh:34,43`, `ab_converge.sh:25,29`, `phase2b_controls.sh:24,40` —
but `$SEED` is only ever interpolated into the **probe** command line. No `BASE` string contains `--seed`,
so every training arm ran at `train_fsdp.py`'s default of **1234** (`train_fsdp.py:715`).

`phase2c_audit.sh:12-14` discovered this for Phase 2b and documented it in a comment, which is good
practice — but the bug was never fixed at the source, the other four scripts still carry it, and
`learnings.md` still attributes seed 0 to the trained checkpoints (L16: "seed 0, patch-8 recipe"; L18:
"global-batch 64, seed 0"). Those attributions are wrong.

Two consequences worth stating plainly. First, the run log for every completed phase records a seed that
was not used, so the record is not reproducible from its own banner. Second, anyone re-running with
`SEED=1 bash scripts/phase2_convstem.sh` gets **bit-identical training** and would reasonably conclude the
result is seed-robust when nothing changed.

**Fix:** add `--seed $SEED` to each `BASE`, and correct the seed attribution in `learnings.md` L16/L18.

### A5 — The H9 hygiene control confounds leakage with training-set size *(Medium)*

`phase2b_h9.sh` trains one conv arm with `--holdout-test-sims` and compares it to the standard conv arm.
But excluding 100 of 1000 sims across all 12 fields removes 18,000 of 180,000 maps
(`train_fsdp.py:486`) — the hygiene arm sees **10% less pretraining data** at the same step count.

The script's stated reading (`phase2b_h9.sh:6-8`) is "heldout < standard ⇒ quantifies the in-suite
optimism". That inference is not available: a deficit is equally explained by the smaller corpus. Only the
null result ("heldout ≈ standard ⇒ leakage negligible") is clean, and only in one direction.

**Fix:** either drop a *matched, random* 100 sims from the standard arm too (so both see 162k maps and only
the identity of the excluded sims differs), or state the confound explicitly and treat H9 as a one-sided
test.

### A6 — Asymmetric model selection between the probe and its baseline *(Medium)*

`train_probe` computes a validation loss every epoch and **prints it** (`probe.py:274-284`). It is never
used — no early stopping, no best-epoch checkpoint. The head evaluated on test is whatever the 20th epoch
produced (`run_probe.py:103`, `probe.py:256`).

`ridge_test` in the baseline, by contrast, *does* use the validation split, selecting the ridge α on it
before reporting test R² (`ps_baseline.py:92-97`) — correctly and with a good docstring.

So the two sides of the headline comparison "SSL probe vs power-spectrum floor" use different protocols:
the classical baseline is tuned on val, the learned probe is not. The 10% val split is being paid for and
then discarded on one side. The direction of the resulting bias depends on whether 20 epochs over- or
under-trains the head, which is not reported anywhere.

**Fix:** select the head by best val loss (three lines), or state that the probe is deliberately
un-tuned and that the comparison is therefore conservative in the baseline's favour.

### A7 — Three different R² definitions *(Medium)*

- `probe.py:327-329` — `ss_tot` from the **test** mean. Standard.
- `ps_baseline.py:99` — `ss_te` from the **test** mean. Standard, matches the probe.
- `rank_report.py:123` — `total = ((y_te - y_tr.mean(0)) ** 2).sum(0)`, the **train** mean.

The third is a different statistic (it scores against a train-mean predictor and is not bounded above by
the usual interpretation). `rank_report.py:213` then prints "Reference: trained attentive probe = Omega_m
0.50" directly beneath its own numbers, inviting a comparison the definitions do not support.

`rank_report.py:107` also fixes `alpha=1e-2` with no selection, where `ps_baseline.py` sweeps 15 values —
another asymmetry between two ridge probes that get compared to each other in the write-ups.

### A8 — Throughput and MFU ignore gradient accumulation *(Medium)*

`train_fsdp.py:639` computes `sps = (n_steps * args.batch * world) / elapsed`, and `estimate_mfu`
(`train_fsdp.py:359`) computes `tokens = args.batch * grid * grid`. Each optimizer step actually processes
`batch * grad_accum` samples per rank (`train_fsdp.py:578-591`). With `--grad-accum 2` both figures are
**understated by 2×**.

`--grad-accum` was added specifically to run the batch-32 A/B (commit `9dc3490`, L11), so any
samples/sec or MFU quoted from those runs is low by the accumulation factor. The headline systems numbers
in the README (`README.md:152-153`) are from `grad_accum=1` runs and are unaffected, but the bug will bite
the next time accumulation is used.

### A9 — Standardization statistics span the test set *(Medium)*

`FieldMapDataset._prepare` (`train_fsdp.py`'s corpus and the probe's dataset alike) accumulates the mean
and std over **every surviving map in the file** (`fields.py:208-227`), then `__getitem__` standardizes
with them (`fields.py:275`). Those two scalars are therefore functions of the test maps, and they are used
to normalize the probe's training inputs.

For 15,000 maps the practical effect on two global scalars is negligible, and this is common practice. It
is worth recording because of where it lands: in the **H9 strict-hygiene arm**, the `Subset` that removes
the test sims (`train_fsdp.py:488`) is applied *after* the manifest and statistics are built from the full
file, so even the arm whose stated property is "the encoder has NEVER seen the test set" (`phase2b_h9.sh:4`)
has seen it through the normalization constants. The claim needs the qualifier.

Related: the disk cache (`fields.py:134-163`) keys on `(n_maps, transform, min_std, H, W)` but **not** on
whether a holdout was applied — correct here, since the cache is built pre-subset, but it is one more
reason the H9 arm's statistics are the full-corpus ones.

### A10 — `rank_report.py` cannot load any current checkpoint *(Medium)*

`ENC = dict(img=256, patch=16, d=1024, heads=16, layers=24)` at `rank_report.py:36` is a module constant
with **no CLI override** (contrast `run_probe.py:84-93`, which added `--patch`, `--enc-d`, `--stem`, …).
Every checkpoint from Phase 0 onward is patch-8, and everything from Phase 2 onward carries `conv_stem.*`
keys. `load_frozen_encoder` will raise on the shape mismatch at `probe.py:75`.

`README.md:147` lists this script in the quickstart. As written it can only be run against pre-Phase-0
checkpoints. Given that the "pooled vs token rank" diagnostic is genuinely interesting and the rank
argument runs through L12–L15, this is worth fixing — it is a copy of the six argparse lines from
`run_probe.py`.

### A11 — CI does not cover the newest stems or the split guards *(Medium)*

- `tests/test_conv_stem.py:89` parametrizes only `["linear", "conv"]`. The `convdisjoint`
  (`jepa_loss.py:241`) and `mlp` (`jepa_loss.py:243`) stems — the S3 and H1 controls, i.e. the arms that
  decide *why* the conv stem works — have no CI gate. `scripts/test_convdisjoint.py` exists but is not
  collected.
- `pytest.ini:2` sets `testpaths = tests`, and the guards under `scripts/` (`test_ps_baseline_split.py`,
  `test_holdout_h9.py`, `test_convdisjoint.py`, `test_conv_stem.py`) expose `main()` rather than `test_*`.
  They are therefore **manual scripts, not gates**, despite guarding the two most leakage-sensitive claims
  in the repo (S1 fairness, H9 hygiene).

The README's correctness-gates section (`README.md:67-80`) is otherwise one of the strongest parts of this
project. These two gaps sit exactly where its own argument says gaps are dangerous.

### A13–A16 — Lower severity

- **`weights_only=False`** at `probe.py:62` unpickles arbitrary objects from a CLI-supplied path. The
  comment explains why (the checkpoint bundles an `args` dict), which is fair, but `weights_only=True` plus
  a separate small metadata read would be safer, and torch's default is moving that way.
- **`src/data/curation.py`** is the Day-3 video path: unused by anything in the CAMELS pipeline, imports
  `decord` at module scope (`curation.py:10`), and resamples rejected clips by **recursive `__getitem__`**
  (`curation.py:59`) — unbounded recursion on a dataset whose clips mostly fail the motion threshold. This
  is the precise pattern `fields.py:5-7` calls out as the thing to avoid. It is also listed in the README's
  engine table as Stage 1 curation (`README.md:18`) alongside `fields.py`, which overstates its role.
  `requirements.txt` correspondingly pulls `transformers`, `decord`, `av`, `einops`, `scikit-learn` for
  what is now a two-file legacy path (`src/infer.py`, `src/data/curation.py`); `decord` in particular is a
  frequent install failure. Consider an `requirements-legacy.txt` split.
- **One mask per batch.** `random_block_mask` (`jepa_loss.py:91`) returns a single `(context_idx,
  target_idx)` pair used for every sample in the step (`train_fsdp.py:148`). I-JEPA samples per-image
  masks. This is a defensible simplification — it keeps the masked forward a clean gather — but it reduces
  mask diversity per step by the batch factor and is nowhere stated as a deviation. `tests/test_masking.py`
  gates structure, not per-sample variation.
- **Two forwards per step.** `_forward_lejepa` runs the encoder on the context subset *and* on the full
  image (`jepa_loss.py:444-447`); the context forward still tokenizes the whole image before gathering
  (`jepa_loss.py:265-271`), so the masking saves transformer compute but not stem compute. The MFU model
  (`train_fsdp.py:366`) charges `6N + 2N` for this, which is approximately right for the transformer and
  slightly optimistic for the conv stem, which runs twice at full resolution. Fine as an estimate; worth a
  note since the conv stem is now the default going forward.

---

## 3. Assumptions the project rests on

These are not defects. They are load-bearing premises that are currently unstated, unenforced, or
unverified — the things that would invalidate results if wrong.

| # | Assumption | Where it bites | Status |
|---|---|---|---|
| P1 | CMD `Maps_*_LH` files store 15 consecutive maps per simulation, so `idx // 15` is the sim id | `fields.py:60-66`, `probe.py:216`, `ps_baseline.py:137` | **Acknowledged as unverified in the code itself** — `fields.py:62`: "Map ORDERING within a sim is still a VERIFY item vs CMD data.py". Every split, every baseline, and the H9 control depend on it. Verifiable offline in minutes against CMD's `data.py`; until then it is the single largest unexamined premise. |
| P2 | `min_std=0.05` rejects **zero** maps in every pooled field, so the manifest is the identity | `run_probe.py:35`, `train_fsdp.py:446`, `probe.py:199` | Argued from `analyze_all` percentiles in a comment (`train_fsdp.py:436-441`) but never asserted at runtime. See A1. |
| P3 | 15 maps of one sim are exchangeable realizations, so map-count inflation of the test set is harmless for point estimates | `probe.py:181` | True for bias, false for variance. Effective n on test is **100 sims**, not 1500 maps. Not stated anywhere. See A2. |
| P4 | Probe-head init noise (±0.03) is the dominant uncertainty | `probe_multiseed.sh:2` | Split variance and pretraining-seed variance are both plausibly larger. S2 (`phase2c_audit.sh`) addresses the second; nothing addresses the first. |
| P5 | The frozen encoder's features are the same every epoch, so bf16 caching is exact | `run_probe.py:45-64` | Holds — augment is off and shuffle is off for the precompute — but the cache stores **bf16** (`run_probe.py:62`, ~3 decimal digits) while the live path feeds fp32. `--no-cache` and cached runs are therefore not numerically identical, and no test pins the difference. |
| P6 | Target standardization constants `(0.3, 0.8) ± (0.1, 0.1)` are close enough to the LH prior | `probe.py:30-31` | Fine — the CAMELS LH priors are U[0.1,0.5] and U[0.6,1.0], giving std ≈ 0.115 — and R² is invariant to it. The code's own comment ("Ideally recompute from the train split") is the right instinct. Harmless, but it *does* affect the ridge α scale in `rank_report.py:204`. |
| P7 | `eff_rank = D / (1 + cov_loss)` (learnings L2) | `sigreg.py:171-173`, `jepa_loss.py:554-559` | Exactly true **only when the covariance diagonal is unit** — then `tr(C)=D`, `‖C‖²_F = D + off_sq`, and the identity follows since `cov_loss = off_sq/D`. The variance hinge enforces that condition, so the relation is real, but it is conditional and the write-up states it unconditionally. |
| P8 | Local-shard var/cov statistics are an adequate substitute for global ones | `sigreg.py:157-159` | Reasonable (B·n ≫ D) and consistent with VICReg. Note the consequence: the objective mixes one **global-batch** term (SIGReg, all-reduced) with three **local-shard** terms (pred, var, cov), so the effective λ balance is world-size dependent in a way that has not been swept. |
| P9 | SIGReg's distributed gradient is "verified equivalent" to single-device | `README.md:19,62`, `sigreg.py:235-240` | Precisely: the **loss** matches exactly; the **gradient** matches up to a factor of `world`, which the DDP/FSDP average then cancels. `tests/test_distributed_sigreg.py:114-120` asserts this correctly and explicitly. `verify_all_reducible` prints the world-scaled case as a **"Warning"** and still exits 0 (`sigreg.py:237`), and the README compresses it to "≡ … in loss *and* gradient". The test is right; the prose and the CLI messaging are looser than the test. |
| P10 | A conv stem is "byte-identical when off", so legacy checkpoints load | `jepa_loss.py:236-245` | Verified by `tests/test_conv_stem.py:45` — good. The assumption that quietly grew is that the **probe's** `--stem` always matches how the checkpoint was trained; a mismatch raises (`probe.py:78`) for structural differences, so this is adequately guarded. |
| P11 | The power spectrum of the **log10** field is a fair stand-in for "the 2-point information ceiling" | `ps_baseline.py:135` | The docstring is honest about this (`ps_baseline.py:21-24`): it answers "what could be extracted from the encoder's input", not "the cosmological P(k)". The README's shorter phrasing — "power-spectrum floor (32-number FFT, no learning)" (`README.md:44`) — loses that qualifier, and a reader will assume the standard density P(k). |
| P12 | The conv-stem gain is the tokenizer, not capacity, periodicity, or nonlinearity | `phase2b_controls.sh`, `phase2c_audit.sh` | This is exactly what H1/H3/H11/S3 are designed to separate, and the design is good. **None of them have reported results in `learnings.md` yet.** The Phase-2 conclusion (L18: "DECISIVE Ω_m lever … band-limiting CONFIRMED and FIXED") is currently supported by a single 2-arm A/B at one pretraining seed with no control arm run. The README is more careful than `learnings.md` here. |
| P13 | Stage-4 inference numbers describe the production encoder | `bench_infer.py:36`, `README.md:21` | `bench_infer.py` benchmarks `ViTEncoder` at **patch-16 / 256 tokens**, the pre-Phase-0 config. The current encoder is patch-8 conv-stem at **1024 tokens** — 4× the sequence, so attention cost scales ~16×. The 3.8×/4.7×/22%-MFU figures are valid for the config they measured and stale for the one now in use. |
| P14 | Velocity fields are magnitudes, so `log10` is safe for all 12 | `train_fsdp.py:436-441` | Supported by the recorded raw-min (+5). But `analyze_all.py:23` still declares `SIGNED = {"Vgas", "Vcdm"}` and analyzes them with `asinh`, while the trainer feeds them `log10`. Two files disagree about the same fields; the trainer's comment is the one with evidence behind it. Reconcile so a future field addition doesn't follow the wrong table. |
| P15 | `_load_cache` invalidation is sufficient | `fields.py:140-160` | Keys on mtime, `n_maps`, `transform`, `min_std`, `H`, `W`. Adequate. Note it does not key on the **code version** of `_transform` — changing the `EPS` floor (`fields.py:28`) or the asinh scale (`fields.py:89`) silently reuses stale statistics. |

---

## 4. Assumptions I made as auditor

Stated so the conclusions can be re-checked:

1. **No GPU, no CAMELS data, no checkpoints.** Everything about training dynamics, R² values, throughput,
   and memory is taken from `learnings.md` and the commit log at face value. I verified *internal
   consistency*, not the numbers themselves.
2. **The `/workspace/logs` outputs referenced by the phase scripts were not available.** Whether H1/H3/H7/
   H9/H11/S2/S3 have actually run is inferred from their absence in `learnings.md` (which ends at L18,
   Phase 2 complete) and from the commit ordering. If those logs exist on a pod, several §3 items resolve.
3. **CMD data layout not checked against upstream.** P1 could be confirmed or refuted in one reading of
   CAMELS' `data.py`; I did not have network access to it and did not want to assert it from memory.
4. **`git log` is treated as the authoritative chronology** where README and `learnings.md` disagree.
5. **I did not run any script that writes to `/workspace`** or that would need data — only `pytest`.

---

## 5. Documentation accuracy

Cross-checks between the three records. Several of these are simply stale; I list them because this repo's
value proposition is the honesty of its numbers, and a reader who spots one drift discounts all of them.

| Claim | Source | Conflicts with |
|---|---|---|
| Current best Ω_m R² **≈ 0.60**, "conv-stem is a hypothesis under test, not a proven win" | `README.md:43,47` | `learnings.md` L18 and commit `900359b` report conv **0.766** and call the verdict "DECISIVE … CONFIRMED and FIXED". The README table was not updated after Phase 2 completed. |
| pk floor Ω_m **0.818** / σ8 **0.331** | `README.md:44`, `phase2b_controls.sh:101`, `learnings.md` L5 | `phase2c_audit.sh:89` quotes the post-S1-fix "fair pk floor (Mgas TEST)" as **0.834 / 0.446**, and `pk+moments` as 0.837/0.544. Commit `15c9d5e` is exactly the fix that changed these. Every table still quoting 0.818/0.331 is pre-fix, i.e. **not** measured on the probe's test sims. This includes the README's headline comparison table. |
| "σ8 already clears the power-spectrum floor (0.388 vs 0.331)" | `README.md:47` | Against the corrected floor of **0.446**, the Phase-1 σ8 of 0.388 does **not** clear it; the Phase-2 conv σ8 of 0.420 does not either. This is the most consequential single discrepancy in the docs — it inverts a headline claim. Worth resolving before anything is shown externally. |
| Phase 2 track 2 "🚧 in progress / next"; results checklist `[ ] Conv-stem A/B` | `README.md:56,158` | Completed at `900529f`/`900358f` (`learnings.md` L18, 2026-07-28) and superseded twice since (Phase 2b, 2c). |
| Trained checkpoints are "seed 0" | `learnings.md` L16, L18 | Training ran at `--seed 1234` (A4). `phase2c_audit.sh:12` documents the discovery. |
| "41 tests" | `README.md:83` | Measured: **42 collected**, 41 passed + 1 skipped on Windows. |
| Distributed SIGReg "world=2 ≡ world=1 … in loss *and* gradient" | `README.md:19,62` | Gradient matches up to ×`world`; see P9. The test file states this correctly, the README does not. |
| `study/` (now the `vjepa-study` repo) depends on `src/`, never the reverse | `README.md` | Holds — verified by grep before the split; no `src/` module imports the study code. ✔ |
| Curation listed as `fields.py` + `curation.py` | `README.md:18` | `curation.py` is the unused video path (A14). The CAMELS curation is entirely `fields.py`. |
| Stage-4 inference levers | `README.md:21` | Measured at patch-16/256 tokens; production is now patch-8/1024 tokens (P13). |

One thing worth saying plainly, since an audit that only lists faults is not an accurate audit: the
qualitative discipline in this repo is unusually good. `learnings.md` records reversals against interest
(L11's false negative, L13/L15 correcting L12, L15 correcting L13's VRAM estimate, L16 correcting the
probe-speed diagnosis); `phase2c_audit.sh:12` documents a bug found in a sibling script rather than quietly
working around it; `ps_baseline.py`'s docstring pre-registers how to read its own result including the
outcome unfavourable to the project; and the pre-registered kill criterion in Phase 1 ("flat/declining ⇒
saturated ⇒ keep n-blocks=4") was written before the sweep and honoured after it. The problems above are
mostly the ordinary decay of a fast-moving research repo, not carelessness.

---

## 6. Recommended order of work

Ranked by (risk removed) ÷ (effort), not by severity alone.

1. **Assert the identity manifest** in `run_probe.py`, `ps_baseline.py`, and the H9 block (A1). ~5 lines.
   Converts the highest-severity silent failure into a crash.
2. **Verify P1** against CAMELS' `data.py` and record the result in `learnings.md`. One reading. Every
   split in the project depends on it.
3. **Fix the R² regex** to `(-?[0-9.]+)` and anchor it to the in-suite block (A3) — or better, have
   `run_probe.py` write a JSON sidecar and delete the three duplicated scrapers. Do this **before** running
   H7, whose result the current scraper cannot read.
4. **Reconcile the pk floor** across README / `learnings.md` / phase scripts to the post-`15c9d5e` numbers,
   and re-examine the σ8 headline against 0.446 (§5). This is a claim-correctness issue, not cosmetics.
5. **Add `--seed $SEED`** to the five orchestrators and correct the seed attribution in L16/L18 (A4).
6. **Add `--split-seed`** and run the linear-vs-conv contrast over 3 splits (A2). Cheap, and it is the
   error bar that the current conclusions actually need — I would prioritize it above the remaining
   H*/S* arms.
7. **Match the H9 arm's data volume** (A5), or restate H9 as one-sided.
8. **Promote the four `scripts/test_*.py` guards into `tests/`** and parametrize the stem tests over
   `convdisjoint`/`mlp` (A11).
9. **Use the val split** for probe model selection, or document the asymmetry (A6); unify the three R²
   definitions (A7).
10. Housekeeping: `grad_accum` in throughput/MFU (A8), `rank_report.py` CLI (A10), split
    `requirements.txt` (A14), the `analyze_all` vs trainer transform table (P14).
