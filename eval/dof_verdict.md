# DOF grounding study — verdict (task #13, suite-sep CORRECTED)

Measured on 2000 maps/suite, Mgas LH, patch sizes {8,16,32}. Pipeline validated:
ITNG pk Omega_m R2 = 0.833 reproduces the Track-1 pk foil (0.834).

⚠ CORRECTION (task #16): the suite-separability row was inflated by scoring the
classifier on its own TRAIN set in 1024-dim. Replaced with 5-fold CV. This also
FLIPS the scale trend (see below). All other rows unchanged.

## Numbers
| quantity                         | ITNG      | SIMBA     |
|----------------------------------|-----------|-----------|
| data-manifold intrinsic dim      | 45-48     | 33-36     |
| participation ratio (64x64 map)  | 42.2      | 29.8      |
| task-relevant DOF (pk k95)       | 10        | 6         |
| pk ceiling R2 (Omega_m / sigma8) | 0.83/0.41 | 0.67/0.28 |
| patch-cloud ID @ P8 / P16 / P32  | 11/16/23  | 10/15/21  |
| suite CV accuracy @ P8/16/32     | 0.59 / 0.63 / 0.65 (was 0.79/0.69/0.66 TRAIN, overfit) |
| suite-dir . cosmology-dir @ P8   | [0.08, 0.12] (near-orthogonal, unchanged) |

## Prescription (grounds the campaign architecture)
- edim ~ 32-64  (manifold ~33-48 dims; only ~6-10 carry cosmology). edim 1024 was
  ~20-30x over-provisioned -> SIGReg spread to rank 283, hoarding nuisance.
  Validate with edim sweep {16,32,64,128,256}; predict transfer peaks ~32-64.
- depth shallow (2-6): task DOF ~6-10 needs little nonlinear mixing (+L21/SKATR).
- per-token dim ~16 suffices (patch-ID@P8 ~11); token edim 1024 was ~60x over.
- patch size: RETRACTED the earlier "feedback lives at small scale" claim -- that
  was the train-overfit artifact. Honest CV separability RISES with patch size
  (0.59@P8 -> 0.65@P32), i.e. the ITNG/SIMBA difference is slightly MORE accessible
  at coarse scale. cosmology (pk) is broadband regardless.

## Cross-suite reads
- cosmology is a LOW-VARIANCE subspace: CCA canonical corr 0.93/0.66 (ITNG) but
  ridge needs whitening to reach it -> variance-equalizing isotropy (SIGReg cov)
  drowns it. Support for alignment-not-isotropy + small edim.
- SIMBA has lower ID and lower pk ceiling than ITNG -> stronger feedback smooths
  structure; ITNG->SIMBA is the harder direction (source richer than target).
- de-classification (#15) DOWNGRADED from GREEN: suites are only WEAKLY separable
  from coarse Mgas features (CV 0.59 @ P8, ~0.09 above chance), so the "remove a
  clean suite direction" premise is weaker than the inflated 0.79 implied. Cosine
  still ~orthogonal to cosmology (low-risk IF pursued). Re-test separability on the
  ENCODER latent (where suite-specific feedback is plausibly more encoded) before
  committing to #15 -- the coarse-feature number is only a lower bound.
