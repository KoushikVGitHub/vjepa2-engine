"""VISReg: Variance-Invariance-Sketching Regularization (Wu, Balestriero & Levine, 2026).

Goal of this file: Implement the *successor* to SIGReg/LeJEPA from the same lab. VISReg keeps
LeJEPA's promise -- a single, principled, negative-free, teacher-free anti-collapse objective --
but fixes the specific failure this repo hit head-on: SIGReg's characteristic-function test goes
gradient-BLIND near anisotropic (dimensional) collapse. We measured it directly (see
study/notes/collapse_resolution.md): at a rank-2 state SIGReg's gradient norm was ~2e-4, so we had
to bolt on a VICReg variance/covariance patch (sigreg.py:variance_covariance_reg) to supply the
missing force. VISReg makes that patch unnecessary by construction.

Why VISReg has a gradient where SIGReg does not
------------------------------------------------
SIGReg matches the empirical CHARACTERISTIC FUNCTION of 1-D projections to that of N(0,1). A
rank-r blob whose total variance is spread so a typical random projection still looks ~unit-Gaussian
passes that CF test with a vanishing gradient, yet has effective rank << D. VISReg instead matches
the SORTED projections (order statistics) to the exact N(0,1) quantiles -- a sliced-Wasserstein
distance to the Gaussian. Order-statistic matching has a well-behaved, non-vanishing gradient
everywhere, including deep in the collapse basin, which is exactly the "generates high gradients
when embeddings collapse" property the paper advertises.

The three terms (all on the SAME embedding batch, no views, no teacher)
-----------------------------------------------------------------------
  center = mean(mu^2)          -- pull the per-dim mean to 0.
  scale  = mean((std - 1)^2)   -- pull each dim's std to 1. TWO-SIDED quadratic, unlike the
                                  VICReg one-sided hinge relu(1-std): it keeps pushing even when
                                  std is healthy-but-not-1, and its gradient does NOT vanish as
                                  std -> 0 (the property the SIGReg-only path lacked).
  shape  = sliced-Wasserstein^2 to N(0,1): project onto K random unit directions, SORT, and match
                                  the sorted values to the theoretical Gaussian quantiles
                                  erfinv(2q-1)*sqrt(2). This is the "sketching" (S) term.

The "Invariance" (I) in VIS is NOT in this module: in a JEPA it is the predictor loss
(pred vs target) supplied by the training loop. This module is a drop-in replacement for
sigreg_loss -- same call site, same (N, D) local-token input -- so `loss = pred_loss +
coef * visreg_loss(full_flat)` is the whole objective (no SIGReg, no var/cov patch, no target-norm
shortcut needed, though target-norm remains compatible).

Distributed note: unlike SIGReg's CF (a pure mean, cleanly all-reducible), the shape term SORTS the
batch, which does not decompose across ranks. Like the VICReg patch we therefore compute VISReg on
the LOCAL shard: with N = batch * n_tokens per rank (e.g. 64*256 = 16384 >> D) the per-rank order
statistics are a fine estimator, and a decorrelation/shape force needs no exact global expectation
the way SIGReg's nonlinear CF did. No rank-synced generator required.
"""
import math

import torch
import torch.nn.functional as F


def visreg_loss(z: torch.Tensor, n_proj: int = 256, eps: float = 1e-6) -> torch.Tensor:
    """VISReg regularizer on a batch of embeddings (local shard).

    Args:
        z: (N, D) embeddings, N = batch * n_tokens on THIS rank. NOT pre-standardized.
        n_proj: number of random 1-D projections for the sliced-Wasserstein sketch (paper K=256).
        eps: std floor, keeps the normalize + the (std-1)^2 gradient finite at exact collapse.

    Returns:
        Scalar = scale + shape + center. 0 iff z is exactly centered, unit-variance per dim, and
        Gaussian-shaped along every projected direction. Its gradient stays strong under
        anisotropic collapse -- the whole reason it replaces the SIGReg + VICReg-patch stack.
    """
    z = z.float()                                          # stats in fp32 (FSDP runs bf16 params)
    N, D = z.shape

    # (1) center: pull the per-dimension mean to 0.
    mu = z.mean(dim=0, keepdim=True)                       # (1, D)
    center = mu.pow(2).mean()
    zc = z - mu

    # (2) scale: pull each dim's std to 1. Two-sided (std-1)^2 -> non-vanishing gradient as std->0,
    #     the exact property SIGReg's marginal CF test lacked against dimensional collapse.
    std = zc.norm(dim=0).div(math.sqrt(N)) + eps           # (D,) per-dim std
    scale = (std - 1.0).pow(2).mean()

    # (3) shape (the "sketch"): sliced-Wasserstein^2 to N(0,1). Normalize with a DETACHED std so the
    #     shape gradient doesn't fight the scale term, project onto K random unit directions, sort,
    #     and match to the exact standard-normal quantiles. Sorting is what makes the gradient
    #     well-behaved deep in the collapse basin.
    zn = zc / std.detach().unsqueeze(0)                    # (N, D), unit-scaled, detached
    W = F.normalize(torch.randn(D, n_proj, device=z.device, dtype=z.dtype), dim=0)  # (D, K) unit dirs
    p_sorted = (zn @ W).sort(dim=0).values                 # (N, K) sorted 1-D projections
    q = torch.linspace(1, N, N, device=z.device, dtype=z.dtype) / (N + 1)           # plotting positions
    target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2)).unsqueeze(1)                 # (N, 1) N(0,1) quantiles
    shape = (p_sorted - target).pow(2).mean()

    return scale + shape + center


if __name__ == "__main__":
    # This module stays a pure library (mirrors sigreg.py). The pedagogical demo -- VISReg keeps a
    # STRONG gradient under dimensional collapse where SIGReg's nearly vanishes -- lives in
    # study/visreg_demo.py.  Run:  python study/visreg_demo.py
    print("Nothing to run here. For the VISReg vs SIGReg collapse-gradient demo: "
          "python study/visreg_demo.py")
