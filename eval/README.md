# eval/ — tool-grounded judge + lineage (thin slice)

The LLM proposes; **tools decide the numbers**; the human decides ship/retract.
Nothing enters `learnings.md` without a tool-grounded, pre-registered verdict.

Design chosen after an external review (verdict: REVISE). Files-as-truth, not a
DB; SQLite is deferred to a rebuildable index; the job queue is deferred.

## Layout
- `lib/trace.py` — **files-as-truth** run store. Each run is an immutable directory
  (`config.resolved.json`, `metrics.json`, `provenance.json`), written temp-then-
  `rename()`; `metrics.json` written last, so its presence == complete. Empirically
  safe on the mfs FUSE volume (single-host concurrency + crash-recovery tested; see
  `scripts/mfs_locktest.py` / `mfs_crashtest.py`). Cross-pod writers untested → keep
  one writer per run.
- `lib/claims.py` — **pre-registered, hash-chained claims** (append-only JSONL, git
  is the tamper-evident ledger). A claim fixes selectors, metric_id, expected range
  or decision rule, and required replicates *before* runs are read. Editing is
  impossible; supersede by `abandon` + a new id. `verify_chain` detects any edit.
- `lib/gate.py` — **deterministic statistics** (no LLM): bootstrap CI (screening
  only), `sd_ucb` (95% upper-confidence run-to-run SD ≈ 4.4× point SD at n=3),
  `combine_sd` (quadrature, not max), `mde`, `decide_comparison`
  (REFUSED/DECISIVE/NULL/INDECISIVE), `floor_parity` (S5), `split_hygiene` (S6).
- `judge.py` — reads a claim + its runs, enforces provenance/parity/hygiene, emits a
  structured verdict. `judge_all` verifies the chain then judges every active claim.
- `test_eval.py` — known-answer tests incl. the **L21 seed-swing trap** (tight 3-seed
  0.79/0.80/0.81 vs 0.72 → INDECISIVE at n=3, not DECISIVE). `python3.13 test_eval.py`.
- `dogfood_dof.py` — runs the DOF findings through end-to-end: CERTIFIES pk-anchor
  (0.833) + data-ID (47.7), **REFUSES suite-separability** because its negative
  control failed (task #16). `python3.13 dogfood_dof.py`.

## Decisions the gate encodes (from the review)
- Noise band = UCB run-to-run SD (S2); report **MDE** every comparison; effects below
  MDE read INDECISIVE — the honest null is institutional, not optional.
- Floor comparison REFUSED unless arm & floor share split/probe-code/HP-budget/
  replicate-count/metric_id (S5).
- Any run whose manifest is non-identity or whose probe split isn't sim-disjoint is
  REFUSED (S6).
- Started-without-terminal runs downgrade a DECISIVE verdict (S8).
- Measurements record `pre_registered` (claim before runs) vs retrospective.

## Not yet built (deferred, on purpose)
SQLite derived index (only if scanning hurts) · submit/worker queue (S12: only if
manual launching becomes the bottleneck) · CI workflow + chain-verify on push ·
cross-validated `suite_sep` metric (task #16) · GPU-arm `trace.log_run` wiring.
