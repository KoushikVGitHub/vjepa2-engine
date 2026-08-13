# TOOLING.md — what actually helps on *this* repo

A practical inventory of the Claude Code skills, agent types, and tools that earn their keep on
**vjepa2-engine**: a from-scratch JEPA/LeJEPA engine (`src/`), driven against CAMELS cosmology
fields, whose real work is **long GPU runs on a rented RunPod pod** orchestrated by `scripts/*.sh`,
**log-scraping into R² tables**, and **CPU correctness gates** (`tests/`, 42 tests, ~11 s, CI on
every push — `.github/workflows/ci.yml`).

The shape of the work matters for tool choice:

- The repo is **Python + torch + bash**. There is no JS/TS, no web app, no server, no issue-driven
  team workflow. Whole families of skills are dead on arrival here.
- The expensive artifact is **GPU-hours on a remote pod**, not local compute. Nothing in this
  toolbox can run a training arm; the tools' job is to *stage it correctly, launch it detached,
  wait without burning tokens, and parse the result faithfully*.
- The scientific record lives in **`learnings.md` (L1…L18)** and **`README.md`**, and the biggest
  recurring failure mode is those two drifting apart. See `CONTEXT_GRAPH.md` for the current drift.

---

## 1. Inventory

### Genuinely useful

| Skill / Agent / Tool | What it does | Concrete use in THIS repo | When to reach for it |
|---|---|---|---|
| **`Bash` with `run_in_background`** | Detached shell that keeps running across turns and re-invokes the agent on exit | The pod loop: `ssh pod 'tmux new-session -d -s p2c "PEAK=165 bash /workspace/vjepa2-engine/scripts/phase2c_audit.sh > /workspace/logs/p2c.log 2>&1"'`, then a backgrounded poller that tails `/workspace/logs/p2c.log` until `=== PHASE 2c DONE ===`. This is the pattern already recorded at `learnings.md:435-438` | Every phase launch. Never run a 7-hour arm in the foreground |
| **`Monitor`** | Blocks on a condition instead of sleeping/polling | Waiting for `grep -q 'PHASE 2b DONE' /workspace/logs/p2b.log` or for a checkpoint file to reach its expected ~844 MB (L15 hit a **truncated 738 MB** ckpt from a disk-quota kill — `learnings.md:204-208`) | Any wait longer than a couple of minutes; strictly better than a `sleep` loop |
| **`TaskCreate` / `TaskUpdate` / `TaskList`** | Background task tracking with status | One task per *arm*, not per phase: `p2b:linear`, `p2b:conv`, `p2b:mlp`, `h9:conv_heldout`, `p2c:convdisjoint`, `p2c:linear_s0`, `p2c:conv_s0`. Phase 2b alone is 3 train arms + 8 probe jobs; without a ledger it is genuinely easy to lose which of H1/H3/H4/H5/H7/H9/H11/S1/S2/S3 has actually reported | Any phase with ≥3 arms — i.e. `phase2b_controls.sh` and `phase2c_audit.sh` |
| **`ScheduleWakeup` / `CronCreate` / `schedule` / `loop`** | One-shot or recurring re-entry | Phase 2 was **~7 h wall** (`learnings.md:303`); Phase 2b is 3 train arms at ~2 h each. Schedule a wake-up at T+2 h to check `nvidia-smi` power draw (see below) rather than idling. `/loop 20m` on a "tail the log and report step/eff_rank" prompt is the cheap babysitter | Runs > 1 h, and any run on a pod you are paying for by the hour |
| **`diagnose` skill** | Reproduce → minimise → hypothesise → instrument → fix → regression-test | Purpose-built for this repo's recurring class of bug: the 2-GPU NCCL hang that shows **100% util at only ~55% TDP** (`learnings.md:105-108`, `learnings.md:197-199`), the `pkill -f train_fsdp` that kills its own SSH shell (`learnings.md:109-111`), the Blackwell `sm_120` "no kernel image" (`learnings.md:250-257`), the OOM-in-backward from SIGReg scaling with token count (`learnings.md:111-113`) | The moment a pod run misbehaves. The "instrument before hypothesising" step is exactly what turned "GPU is busy" into "GPU is idle at 76 W" |
| **`tdd` skill** | Red-green-refactor | The repo already works this way and it pays: `--stem conv` shipped with `tests/test_conv_stem.py` *before* the A/B (`844916c`), `--holdout-test-sims` shipped with `scripts/test_holdout_h9.py` (`fc2470d`), `convdisjoint` with `scripts/test_convdisjoint.py` (`4681b87`). Write the CPU test that pins the *geometry* first, then the flag | Every new `--stem` / `--probe-stage` / split-affecting flag. A silently wrong control costs a 2-hour GPU arm |
| **`grill-me` skill** | Dependency-ordered interrogation of a plan until every branch resolves | Already load-bearing here: the entire Track-3 design (value prop → sequencing → masking → exit gate, `learnings.md:308-372`) came out of a grill-me session, and `learnings.md:449-450` records it. The next natural target is the Phase-3 SIMBA transfer design (normalization choice, retention metric, what counts as a win) | Before committing GPU-hours to a new *phase*, not a new arm |
| **`Explore` agent** | Read-only fan-out search, returns conclusions not file dumps | "Where is the pk floor number hard-coded?" returns 20 hits across `README.md`, `learnings.md`, and six `scripts/*.sh` — exactly the sprawl that made the S1 correction (0.818 → 0.834) hard to propagate | Cross-cutting questions over ~15 markdown/shell files where you want the answer, not the excerpts |
| **`general-purpose` / `Plan` agents** | Multi-step research; implementation planning | `Plan` for "add hypothesis arm S4 end-to-end" (touches `src/jepa_loss.py`, `src/train_fsdp.py` argparse, `scripts/run_probe.py`, a `phase*.sh`, and a CPU test). `general-purpose` for audits like the one that produced `CONTEXT_GRAPH.md` | Work that spans ≥4 files or needs a written trace |
| **`write-a-skill`** | Authors new skills with progressive disclosure | See §3 — this repo has three recurring, mechanical, error-prone rituals that are perfect skill material | Once a workflow has been done by hand ≥3 times (phase-report scraping is at 5+) |
| **`dataviz` skill** | Chart/plot design system before you write chart code | The R²-vs-steps convergence ladder (`ab_converge.sh` probes at 2000/4000), the mask-ratio curve (`learnings.md:227-232`), the stem A/B bar with the pk floor as a reference rule, the eff_rank-vs-global-batch scatter that L12/L13/L15/L18 disagree about. All of these are currently ASCII tables | Any time a result would read better as a figure — especially the eff_rank-vs-batch plot, which is where four learnings contradict each other |
| **`Artifact` + `artifact-design`** | Publish a private, self-contained hosted page; same URL on redeploy | Already in the workflow (`learnings.md:446`). The right artifact is a **phase dashboard**: the decision ledger + numbers-of-record from `CONTEXT_GRAPH.md`, the R² tables, and the H*/S* status board, redeployed at each phase close | End of a phase, or when handing state to a collaborator |
| **`WebSearch` / `WebFetch`** | Read the actual source instead of recalling | The rule is already written down (`learnings.md:439-442`): confirmed VISReg's `num_projections=4096` and that Galaxy10 is downstream-only. Live needs: the CMD `o3_err` supervised baseline numbers, the published CAMELS VAE linear-probe R² 0.93 caveat (`the vjepa-study repo (notes/collapse_resolution.md):124-129`), CAMELS SIMBA suite download specifics for Phase 3 | Any claim about external literature or a dataset you are about to cite in README |
| **`init` (CLAUDE.md)** | Generate a repo CLAUDE.md | **This repo has no CLAUDE.md.** The non-obvious invariants (loss going up = healthy; `eff_rank` not `tgt_std`; split must be sim-level; probe `--stem` must match the ckpt) are currently only discoverable by reading 37 KB of `learnings.md` | Do this once, soon; seed it from `CONTEXT_GRAPH.md` §5 |
| **`fewer-permission-prompts`** | Scans transcripts, writes a scoped allowlist into `.claude/settings.json` | This session's traffic is dominated by `git log`, `pytest`, `grep`, and (on pod days) `ssh pod '…'` and `scp`. There is currently **no `.claude/` directory** in the repo | Once, early. It pays back every pod session |
| **`update-config`** | Edit `settings.json`, permissions, env vars, hooks | Pin `PYTORCH_KERNEL_CACHE_PATH` guidance, allowlist the pod SSH host, and add a **Stop hook that runs `pytest -q`** so the CPU gates cannot silently rot between commits | When you want an automated behaviour (hooks), not a remembered preference |
| **`/code-review`, `review`, `security-review`** | Multi-agent review of the working diff / a PR | A review agent already caught 3 real bugs here including *a post-abort save that would have destroyed a checkpoint* (`learnings.md:451-452`). Highest-value target: any diff touching `sim_split`, `--holdout-test-sims`, or the probe/pk split alignment — a leak there invalidates every number in the repo | Before pushing anything that changes a split, a loss term, or a checkpoint write path |
| **`simplify`** | Quality pass over changed code (reuse, altitude), not bug-hunting | The six `phase*.sh` scripts have heavily duplicated `BASE=` flag strings, `probe_ms()`, and `agg()` heredocs (compare `phase2b_controls.sh:49-74` with `phase2c_audit.sh:43-68` — near-identical). One shared `scripts/_lib.sh` would remove a whole class of "the arms weren't matched" bug | After a phase closes, before the next one forks the script again |
| **`edit-article`** | Restructure and tighten prose | `README.md` is the public artifact and is currently **stale by a full phase** (see `CONTEXT_GRAPH.md` §3 Discrepancies). `learnings.md` has L13 printed before L12 (`learnings.md:141` vs `:160`) | The "turn a completed phase into docs" workflow (§2e) |
| **`git-guardrails-claude-code`** | Hooks that block destructive git commands | Work flows **local → push to `main` → pod pulls** (`learnings.md:445`). A `reset --hard` or force-push here desyncs the pod mid-phase | Worth 5 minutes once |
| **`design-an-interface`** | Parallel sub-agents propose several module shapes | Exactly one pending API decision is worth it: **T5**, injecting external mean/std into `FieldMapDataset` for the zero-shot SIMBA normalization (`learnings.md:338-340`, `:366-367`). The choice determines whether the headline transfer number is honest | When you reach Phase 3, not before |

### Not useful here, and why

| Skill/Tool | Why it does not apply |
|---|---|
| `migrate-to-shoehorn` | TypeScript-only (`as` assertions, `@total-typescript/shoehorn`). This repo has zero TS. |
| `scaffold-exercises` | Builds course exercise directories. the `vjepa-study` repo is a personal study log, not a course. |
| `setup-pre-commit` | Husky + lint-staged + Prettier — a Node toolchain. If pre-commit hooks are wanted here, use `update-config` with a Python/pytest hook instead. |
| `claude-in-chrome` | No browser surface. The only remote surface is SSH to a pod. |
| `claude-api` | The repo contains no LLM/Anthropic-SDK code — it is torch + numpy. The skill's own SKIP rule ("another provider being worked on") doesn't even fire; there is simply no LLM call site. |
| `obsidian-vault` | Notes live in `the vjepa-study repo (notes/*.md)` in-repo and are read by CI-adjacent docs links, not an Obsidian vault. |
| `keybindings-help` | Editor ergonomics, orthogonal to the work. |
| `run` | Its job is "launch this project's app and screenshot it". This project's "app" is a 2-GPU 7-hour `torchrun` on a rented pod; none of the skill's built-in patterns (CLI/server/TUI/Electron/browser) match, and guessing wrong wastes GPU time. Use the `phase*.sh` scripts. |
| `statusline-setup` agent | Cosmetic. |
| `to-prd`, `to-issues`, `triage`, `qa`, `request-refactor-plan` | All assume an issue-tracker-driven team workflow. This is a single-author repo whose backlog lives in `learnings.md` and the `phase*.sh` headers, and whose "issues" are hypotheses (H1…H11, S1…S3) with pre-registered read rules. `to-issues` becomes marginally useful *if* the H*/S* battery is ever moved onto GitHub Issues — it is not today. |
| `improve-codebase-architecture` | Explicitly keyed to `CONTEXT.md` + `docs/adr/`, neither of which exists. The equivalent record here is `learnings.md`. `simplify` covers the real need (the duplicated `phase*.sh` bodies). |
| `caveman` | Fine for chat compression, but this project's value *is* the written reasoning trace. Compressing it fights the goal. |
| `Artifact` capabilities (live data / shared state) | The published pages are static result reports. No runtime capability is needed; declaring one adds surface for nothing. |

---

## 2. Workflows

### (a) Launching a long GPU arm on the pod, and picking the result back up

The invariant: **runs are launched by `scripts/*.sh` inside `tmux` on the pod, and every arm writes
to `/workspace/logs/`.** Nothing about that needs to change; the tooling wraps it.

1. **Sync first.** `git push` locally; on the pod `cd /workspace/vjepa2-engine && git pull`. Every
   `phase*.sh` does `cd $WS/vjepa2-engine` (e.g. `phase2c_audit.sh:16`), so a stale pod checkout
   silently runs the *old* flag set — the single most expensive mistake available.
2. **Preflight, cheap, before you burn 2 hours.** These are all L-recorded failures:
   - `df -h /workspace` — L15 lost the final step-4000 checkpoint to a quota kill and had to fall
     back to step-3000 (`learnings.md:204-208`). 4 × 844 MB ckpts + a 53 GB corpus is tight under 75 GB.
   - `nvidia-smi --query-gpu=name,memory.total --format=csv` — batch 48/GPU needs 24 GB+;
     20 GB cards force BATCH=32 (`learnings.md:303`), 16 GB A4000s cannot train patch-8 at all
     (`phase2_convstem.sh:18-19`).
   - `which tmux || apt-get install -y tmux` (`learnings.md:191`), and on Blackwell (sm_120) the
     torch cu128 upgrade (`learnings.md:250-257`).
   - `PEAK=` must match the card or the logged MFU is fiction: 165 (4090), 77 (A4000), 153 (RTX 4000 Ada), 312 (A100-class).
3. **Launch detached.** Never `nohup … &` over a one-shot SSH — the job races SIGHUP and dies with
   no log (`learnings.md:114-115`). Always:
   `ssh pod 'tmux new-session -d -s p2c "PEAK=165 bash /workspace/vjepa2-engine/scripts/phase2c_audit.sh > /workspace/logs/p2c.log 2>&1"'`
4. **Register the arms.** `TaskCreate` one task per arm. `phase2c_audit.sh` has three train arms and
   nine probe jobs (3 arms × 3 head-seeds).
5. **Wait without burning tokens.** `Monitor` on `ssh pod "grep -c '=== PHASE 2c DONE ===' /workspace/logs/p2c.log"`,
   plus a `ScheduleWakeup` at T+90 min for a **health check** — and the health check is
   `nvidia-smi --query-gpu=power.draw,utilization.gpu`, because **power draw, not utilization, is the
   NCCL-hang diagnostic** (100% util at ~55% TDP means hung; `learnings.md:105-108`, `:197-199`).
6. **Pick it up.** `scp pod:/workspace/logs/p2c*.log` into the scratchpad, then parse (§b). If a run
   died, `diagnose` — and check the checkpoint *size*, not just its existence.

### (b) Parsing probe logs into result tables

Every script already ends with an in-line aggregator (`phase2b_controls.sh:57-74`,
`phase2c_audit.sh:51-68`, `probe_multiseed.sh:27-45`) that regexes
`R2\s*:\s*Omega_m=([0-9.]+)\s+sigma8=([0-9.]+)` out of `/workspace/logs/*.log` and prints
`mean ± pstdev` over head-seeds. The manual workflow is:

1. `scp` the whole `/workspace/logs/` for the phase into the scratchpad (never edit logs on the pod).
2. Re-run the aggregation **locally** with `Bash` over the copied logs — the heredocs are pure Python
   and need no torch — so the numbers in the report are reproducible from files you still have.
3. Emit a markdown table with the **provenance columns** the ledger needs: script, global batch,
   steps, pretraining seed, probe head-seeds, checkpoint filename, field, split. Bare R² numbers are
   how `README.md` and `learnings.md` drifted apart in the first place.
4. Convert to a figure with `dataviz` if the result is a *curve* (steps, mask ratio, batch) rather
   than a 3-row A/B.
5. Two traps, both real:
   - `probe_multiseed.sh:23` greps `-A2 "IN-SUITE"`, so a probe that also ran the cross-suite
     (SIMBA) block still reports the in-suite number — but only because the in-suite header comes
     first (`scripts/run_probe.py:159` then `:162+`). Do not reorder those prints.
   - `std` here is `statistics.pstdev` over **probe-head seeds only**. It is *not* an error bar on
     the science (see §c and `phase2c_audit.sh:9-10`).

### (c) Adding a new reviewer-control arm (an "H"/"S" hypothesis), end to end

The repo has a consistent five-part shape for a control. Follow it exactly; `4681b87` (S3
`convdisjoint`) is the cleanest reference commit.

1. **Write the hypothesis and its read rule first**, as a header comment, before any code. Every
   script does this: `phase2b_h9.sh:4-8` ("heldout ≈ standard ⇒ leakage negligible; heldout <
   standard ⇒ quantifies the optimism"), `phase2c_audit.sh:5-10`. A control without a
   pre-registered read is post-hoc.
2. **Model/flag change in `src/`.** New stem variants go in `src/jepa_loss.py` (`ConvStem` gained
   `overlap={True,False}` for S3); the flag is threaded through `src/train_fsdp.py`
   (`--stem` choices at `src/train_fsdp.py:735`) *and* `scripts/run_probe.py:89` — the probe stem
   **must** match the checkpoint's, and nothing enforces that at load time.
3. **CPU test that pins the property the control depends on.** Use `tdd`. The existing three:
   - `scripts/test_convdisjoint.py` — perturb one interior patch, assert `convdisjoint` changes
     exactly 1 token while `conv` changes the token *and* its neighbours (GroupNorm swapped to
     Identity to isolate conv geometry).
   - `scripts/test_holdout_h9.py` — pure index arithmetic: the pretraining exclusion set equals the
     probe's test sims, across all 12 fields, exactly.
   - `scripts/test_ps_baseline_split.py` — `ps_baseline.py`'s split is index-identical to
     `probe.sim_split`.
   ⚠ **All three live in `scripts/`, which is outside `pytest.ini`'s `testpaths = tests`, so CI
   never runs them.** Only `tests/test_conv_stem.py` was ported. Port new control tests into
   `tests/` (or extend `testpaths`) or they rot silently.
4. **A `phase*.sh` that runs it matched.** Copy the `BASE=` string verbatim from the phase you are
   controlling against — `phase2c_audit.sh:22-24` matches `phase2b_controls.sh:28-30` flag for flag.
   Any unmatched flag confounds the arm.
5. **Update the ledger.** New row in `CONTEXT_GRAPH.md` §2 with status `hypothesis / not yet run`,
   and a new `L{n}` in `learnings.md` when it reports.

**Seed discipline — read this before adding an arm.** `src/train_fsdp.py:715` defaults
`--seed 1234`. Several `phase*.sh` define a shell `SEED=0` and echo "seed 0" but never pass `--seed`
into the training command (`phase2_convstem.sh:34` + `:39-41`; `phase2b_controls.sh:24` + `:28-30`).
`phase2c_audit.sh:12-14` catches this explicitly and says the phase2b comment "is WRONG". So the
shell `SEED` variable reaches only the **probe head**, and every "seed 0" pretraining claim in
`learnings.md` is actually seed 1234. If your arm needs a genuinely different pretraining seed, pass
`--seed` inside the flag string, as `phase2c_audit.sh:29-30` does.

### (d) Keeping the CPU correctness gates green

- Local: `pip install -r requirements-dev.txt && pytest` — 42 collected tests, CPU-only, no CAMELS
  data, no `transformers`. (`README.md:83` says "41 tests"; the cache and a fresh collect say 42.)
- CI: `.github/workflows/ci.yml` on push to `main`, PRs, and manual dispatch; Python 3.10 + 3.12,
  CPU torch wheel only, and it **re-runs `tests/test_distributed_sigreg.py` as its own step** because
  that gloo world=2 ≡ world=1 invariant is the repo's load-bearing correctness claim.
- Windows note: the gloo test skips locally (gloo cannot create a device on Windows) — a green local
  run is **not** a green CI run. Push and check the badge, or run under WSL.
- Torch's CPU build **segfaults on the true 24-layer attention**, which is why every test caps depth
  at ≤2 layers (`tests/conftest.py` docstring, `tests/test_conv_stem.py` docstring). Do not "fix"
  a CI hang by adding depth.
- Use `diagnose` for a red gate; use `/code-review` before pushing a diff that touches `sigreg.py`,
  `sim_split`, or masking, since all three fail *silently* rather than loudly.
- Suggested `update-config` hook: run `pytest -q` on Stop, so a session cannot end with a red suite.

### (e) Turning a completed phase into README / learnings updates

This is the workflow that is currently *behind*: commit `900359b` ("Phase 2 COMPLETE") touched
**`learnings.md` only** — 15 lines added, no README change — and the later `57995a0` edited README
for CI/licence without refreshing the results. So `README.md:56` still calls Phase 2 track 2
"🚧 in progress / next" and `README.md:43` still headlines Ω_m ≈ 0.60.

The ritual that would have caught it:

1. **`learnings.md` first** — a new `L{n}` with: the arms, the exact flags/batch/steps/seed, the
   result table, the verdict against the *pre-registered* read rule, the infra gotchas, and what the
   next phase inherits. L18 (`learnings.md:291-304`) is the model.
2. **Then propagate to `README.md`** in all four places it duplicates numbers: the probe table
   (`:41-45`), the prose verdict (`:47`), the phase ladder (`:51-58`), and the results checklist
   (`:150-160`). Use `Explore` to find every literal of the number you are changing — the pk floor
   `0.818` alone appears **20 times** across README, learnings, and six shell scripts.
3. **Then `phase*.sh` reference lines** — the scripts echo the comparison bar for the *next*
   operator (`phase2b_controls.sh:101`, `phase2_convstem.sh:65`, `mask_sweep.sh:58`). A stale bar
   there means the next run is read against the wrong floor.
4. **`edit-article`** on the README diff for prose; **`dataviz`** if the phase produced a curve;
   **`Artifact`** to redeploy the phase dashboard at the same URL.
5. **Commit message states the verdict and the number**, matching the repo's existing style
   (`900359b`: "conv-stem lifts Omega_m 0.55->0.77 (+40%), band-limiting confirmed & fixed").

---

## 3. Gaps — skills worth authoring with `write-a-skill`

Three rituals here are frequent, mechanical, and have already produced real errors. None is covered
by an existing skill. (Proposals only — not created.)

### 3.1 `phase-report` — scrape `/workspace/logs` into a provenance-carrying results table

- **Trigger:** "phase 2c finished", "parse the p2b logs", "what did the h9 arm say", `/phase-report`.
- **Inputs:** a phase tag (`p2b`, `p2c`, `h9`, `ms`, `cv`) or a local log directory; optional
  `--seeds`, `--field`, and a `--baseline` row to compare against.
- **Steps:**
  1. Locate logs — local dir, or `scp -r pod:/workspace/logs/${tag}*` into the scratchpad. Never
     parse over a live SSH pipe; keep the artifact.
  2. Parse `R2\s*:\s*Omega_m=…\s+sigma8=…` per file, grouping by arm label and head-seed, mirroring
     the aggregators at `phase2b_controls.sh:57-74` / `phase2c_audit.sh:51-68`.
  3. Recover **provenance** from the sibling `*_train_*.log` and the script itself: global batch
     (`batch × nproc`), steps, `--stem`, `--stem-pad`, `--probe-stage`, checkpoint path, and the
     **actual pretraining seed** (default 1234 unless `--seed` appears in the command line — the
     exact trap `phase2c_audit.sh:12-14` documents).
  4. Emit a markdown table: arm | Ω_m ± head-std | σ8 ± head-std | n seeds | global batch | steps |
     pretrain seed | ckpt. Label the ± explicitly as **probe-head-init only**.
  5. Read the arms against the script header's pre-registered rule (`phase2b_controls.sh:102-103`)
     and state which hypotheses the run closes and which it leaves open.
  6. Flag anomalies: `train rc≠0`, fewer step-logs than `steps/log-every`, a checkpoint under
     ~800 MB (the L15 truncation), or `eff_rank` below the 32-dim floor.
  7. Offer the `learnings.md` `L{n}` block and the `README.md` line-by-line diff as output.
- **Why a skill:** done ≥5 times by hand (Phase 0, 1, 2, 2b, 2c) and the provenance columns — the
  ones that prevent doc drift — are exactly what gets dropped when it is done ad hoc.

### 3.2 `new-control-arm` — scaffold an H*/S* hypothesis across flag + script + test

- **Trigger:** "add a control for X", "new hypothesis arm", "H12 / S4", `/new-control-arm`.
- **Inputs:** hypothesis id and one-line claim; the **pre-registered read rule** (what result means
  what — refuse to proceed without it); which layer it touches (stem / masking / split / probe
  stage); the phase whose `BASE=` it must match.
- **Steps:**
  1. Write the header comment block first — claim, read rule, and what it forecloses — in the
     `phase2b_h9.sh:2-10` house style.
  2. Add the flag: `src/train_fsdp.py` argparse near `:706-745`, threaded to `build_model`; mirror
     it in `scripts/run_probe.py:89-99` if the probe must reconstruct the same architecture. Assert
     the default path stays **bit-identical** (the `--stem linear` guarantee at `learnings.md:274-275`)
     so legacy checkpoints still load.
  3. Generate a CPU test in **`tests/`** (not `scripts/` — CI does not see `scripts/`) that pins the
     one geometric or index property the control rests on, patterned on
     `scripts/test_convdisjoint.py` / `scripts/test_holdout_h9.py`.
  4. Generate the `phase*.sh` by copying the target phase's `BASE=` string verbatim and appending
     only the new flag; wire the standard `probe_ms` / `agg` / `run_paired` helpers; end with the
     `reads |` echo line.
  5. Emit the exact `tmux new-session -d` launch line with the right `PEAK=` for the card.
  6. Append a `hypothesis / not yet run` row to `CONTEXT_GRAPH.md` §2 and a stub `L{n}`.
- **Why a skill:** five arms were added in four days (`36609ad`, `fc2470d`, `4681b87`, `27b3bda`) and
  the one thing that varies between them — which flags are matched — is the one thing that
  invalidates the arm if it slips.

### 3.3 `numbers-audit` — keep README, learnings, and the scripts telling the same story

- **Trigger:** "check the docs agree", before any push touching `README.md`/`learnings.md`, or after
  a baseline is recorrected (as S1 did to the pk floor).
- **Inputs:** none by default; optionally a specific metric to trace.
- **Steps:**
  1. Extract every headline number with its file:line from `README.md`, `learnings.md`,
     `the vjepa-study repo (notes/*.md)`, `scripts/*.sh` echo lines, and commit subjects (`git log --pretty=%s`).
  2. Cluster by metric identity — pk floor Ω_m, pk floor σ8, best conv Ω_m, best σ8, eff_rank vs
     global batch, encoder param count, test count — and report every cluster with >1 distinct value.
  3. Check derived claims still hold under the newest inputs. The live example: the σ8 headline
     "0.388 > 0.331, clears the power-spectrum floor" (`README.md:47`) is asserted against the
     **pre-S1** floor, while `phase2c_audit.sh:89` now quotes a fair floor of **σ8 0.446** — above
     the best measured σ8 of 0.420. A recorrected baseline can silently invert a headline.
  4. Verify each number's provenance is still reachable: the script, the flags, the checkpoint name,
     the split.
  5. Output a discrepancy table with both sources and line refs. **Never auto-resolve** — a
     discrepancy is usually a stale doc, but sometimes it is a real unrecorded experiment.
- **Why a skill:** `CONTEXT_GRAPH.md` §3 found eight live discrepancies in a repo whose entire value
  proposition is "logged numbers, nothing overstated" (`README.md:12`, `:47`).
