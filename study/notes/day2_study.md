# Day 2 — Inside JEPA + SIGReg (study + build log)

## Pre-test debrief (cold scores)
- P1 forward pass / stop-grad: 4/5 — had modality-agnostic x/y encoders + stop-grad on
  target branch; fixed: predictor is on the CONTEXT branch, stop-grad sits between
  y-encoder output and the loss (target = frozen label, EMA-updated only).
- P2 EMA + what breaks w/o stop-grad: 1.5/5 (the rust). EMA = Exponential Moving Average.
  Without stop-grad -> COLLAPSE: optimizer drives the *target* encoder to a constant to
  zero the loss; both sides race to a degenerate fixed point. Stop-grad makes the target a
  fixed label so the only way down is real prediction. EMA = stable-but-improving target.
- P3 latent space + smooth-L1: 4/5. smooth-L1 = L2 near 0 (smooth) + L1 in tails (robust to
  noisy EMA targets). Non-contrastive, non-generative -> constant-vector shortcut is why
  collapse is even a danger.
- P4 mask ratio -> 100%: 4/5. No context -> nothing to predict from (starvation collapse).
  Quality is NON-monotonic in mask ratio: too low = solvable by interpolation; sweet spot =
  aggressive block masking (high but < total).
- P5 SIGReg + Cramer-Wold: 3/5. Pushes embeddings -> isotropic Gaussian. Sharpened: it's a
  DISTRIBUTIONAL objective (no pairs; not "push points apart"). Two parts: cov ~= I
  (decorrelated axes, equal var) + max-entropy Gaussian shape. Collapse is maximally
  non-Gaussian -> repelled by construction -> no EMA/stop-grad needed. Cramer-Wold: N(0,I)
  iff every 1-D projection is N(0,1); test random projections via characteristic-function
  normality test, sum. Linear, ~50 lines, distributed (all-reduce batch stats). Payoff:
  linear identifiability (latent factors up to rotation) = cleaner control/interp surface
  = the safety hook.

## SIGReg sanity demo (src/sigreg.py)
    N(0,I) target   0.00008   ~0           target distribution        OK
    collapsed       0.16110   large        repelled by construction   OK (anti-collapse proof)
    scaled var=25   0.30438   largest      wrong variance penalized    OK
    uniform         0.00010   ~0           expected "small-nonzero"   SURPRISE -> investigated

## The uniform surprise -> ablation (analysis/sigreg_freq_dim_ablation.py)
Eval loop: printed expectation said the toy demo passed, but the `uniform` number was
suspiciously low. Interrogated it.
- Hypothesis 1 (frequencies): t ~ N(0,1) samples mostly low freq; maybe we miss the
  high-freq kurtosis difference. FALSIFIED — scaling t up did NOT move uniform, and it made
  `scaled var=25` score LOWER (variance errors are detected at LOW freq; all CFs decay to 0
  at high t, Riemann-Lebesgue).
- Hypothesis 2 (dimension/CLT): random 1-D projections of high-dim data are ~Gaussian
  (Diaconis-Freedman). CONFIRMED — uniform scores ~160x the Gaussian control at D=1 and
  decays monotonically to ~1x by D=64.

### Limitation for the writeup
SIGReg (and any Cramer-Wold / 1-D-marginal test) reliably catches COLLAPSE and ANISOTROPY
(both show up in projections as wrong variance / low-rank directions — confirmed by the
collapsed & scaled cases). It is BLIND to per-axis non-Gaussianity that survives only in the
high-dim joint, because random projection Gaussianizes the marginals it tests. "Passes
SIGReg" == "anti-collapse + isotropy satisfied", which is the property the objective needs —
NOT "Gaussian in every sense". State this honestly; it's the failure-envelope understanding
that separates a real reproduction from a fan's.

## BUILD result — what ACTUALLY prevents collapse (src/jepa_loss.py)
Built tiny ViT JEPA (block masking + context/target encoders + predictor + EMA). First run
tried to show "decay=0 -> collapse" and FAILED — every decay stayed healthy. The data
falsified the morning claim. Corrected experiment (stop_grad switch):
    no stop-grad (symmetric)   loss 0.00007  std 0.0095  COLLAPSED
    stop-grad, decay=0.0       loss 0.27     std 0.81    healthy
    stop-grad, decay=0.998     loss 0.32     std 0.82    healthy
LESSON (corrects Q1): the STOP-GRADIENT (+ predictor) is the load-bearing anti-collapse
mechanism, NOT the EMA decay. This is SimSiam (Chen & He 2020): stop-grad alone prevents
collapse; the momentum/EMA encoder is a quality/stability booster, not strictly required.
"decay=0 -> target==online -> collapse" is wrong WHEN stop-grad is present: each step the
target is a fixed (no_grad) label, so nothing rewards constancy. Collapse needs the
asymmetry REMOVED (symmetric, grad into both sides) -> then constant vector is the global min.
Meta: the diagnose loop fired on ME — predicted result, run, falsified, corrected.

## Still TODO Day 2 (fresh head, tomorrow)
- [x] Read LeJEPA (arXiv 2511.08544) SIGReg section — DONE (Koushik read full paper).
- [x] BUILD: masking + tiny ViT + predictor + collapse experiment — DONE (src/jepa_loss.py).
- [ ] LITMUS: derive JEPA loss on whiteboard, COLD, no AI. <-- only remaining Day-2 item.
- [ ] Code walkthrough: Claude runs Koushik through src/jepa_loss.py line-by-line
      (patchify -> encoder/keep -> predictor mask-tokens -> stop-grad branch -> collapse sweep).
      Do AFTER the litmus (so the derivation is cold, not primed by the walkthrough).
