"""Day 2 (★) - SIGReg from scratch: Sketched Isotropic Gaussian Regularization.

This is the core of LeJEPA (Balestriero & LeCun, 2025, arXiv 2511.08544): replace the
EMA-teacher + stop-grad anti-collapse *heuristics* with ONE principled objective -
push the embedding distribution toward an isotropic Gaussian N(0, I).

Why it can't collapse: an isotropic Gaussian is the maximum-entropy distribution for a
fixed variance, so it carries maximal information. A collapsed (constant / low-rank)
embedding is maximally *non*-Gaussian, so the regularizer repels it.

How it scales (Cramer-Wold): a distribution is N(0, I) IFF every 1-D projection of it is
N(0, 1). So we sketch: draw random unit directions, project embeddings onto each, and test
each 1-D projection against N(0, 1). Many cheap 1-D tests == one high-D test, in linear time.

The 1-D test uses the characteristic function (CF). The CF of N(0,1) is exp(-t^2/2). We
compare the empirical CF of the projected samples to that target over a set of frequencies
t; the squared difference, weighted/averaged, is an Epps-Pulley-style normality statistic.

NOTE: this is faithful *in spirit*; the official LeJEPA repo (github.com/rbalestr-lab/lejepa)
uses a specific closed-form statistic. Compare against it during study.

Distributed note: the regularizer is an expectation over the batch (means of cos/sin over
samples), so at multi-GPU scale you all-reduce the per-device partial sums to get the
global-batch statistic -- no negative-pair gathering. That's why SIGReg is distributed-friendly.
"""
import torch
import torch.nn.functional as F


def random_directions(dim: int, n_proj: int, device, generator=None) -> torch.Tensor:
    """`n_proj` random unit vectors in R^dim, shape (dim, n_proj). The 'sketch'."""
    v = torch.randn(dim, n_proj, device=device, generator=generator)
    return F.normalize(v, dim=0)


def sigreg_loss(z: torch.Tensor, n_proj: int = 256, n_freq: int = 64) -> torch.Tensor:
    """SIGReg regularizer: distance from N(0, I) via sketched 1-D CF tests.

    Args:
        z: (B, D) batch of embeddings (NOT pre-standardized -- the loss itself pulls
           mean -> 0 and variance -> 1).
        n_proj: number of random 1-D projections (the sketch dimension).
        n_freq: number of CF frequencies sampled per projection.

    Returns:
        scalar loss; 0 iff the projected embeddings are exactly N(0, 1) in every direction.
    """
    B, D = z.shape
    V = random_directions(D, n_proj, z.device)        # (D, n_proj)
    P = z @ V                                          # (B, n_proj)  projected samples

    # Frequencies sampled from N(0,1): Monte-Carlo of the Gaussian-weighted CF integral.
    t = torch.randn(n_freq, device=z.device)          # (n_freq,)
    tp = P.unsqueeze(-1) * t                           # (B, n_proj, n_freq)

    # Empirical characteristic function of each projection: E[e^{i t P}] = E[cos] + iE[sin].
    cos = tp.cos().mean(dim=0)                         # (n_proj, n_freq)
    sin = tp.sin().mean(dim=0)                         # (n_proj, n_freq)

    target = torch.exp(-0.5 * t**2)                    # CF of N(0,1): real, (n_freq,)

    # Squared CF distance: real part should match exp(-t^2/2), imaginary part should be 0.
    loss = ((cos - target) ** 2 + sin ** 2).mean()
    return loss


if __name__ == "__main__":
    # Sanity demo: SIGReg should be ~0 for true N(0,I), and LARGE for collapsed/structured data.
    torch.manual_seed(0)
    B, D = 4096, 64

    gaussian = torch.randn(B, D)                       # the target distribution
    collapsed = torch.zeros(B, D) + 0.01 * torch.randn(1, D)  # near-constant == collapse
    scaled = 5.0 * torch.randn(B, D)                   # right shape, wrong variance
    uniform = (torch.rand(B, D) - 0.5) * 3.46          # zero-mean, ~unit-var, but NOT Gaussian

    for name, x in [("N(0,I) target", gaussian),
                    ("collapsed", collapsed),
                    ("scaled var=25", scaled),
                    ("uniform", uniform)]:
        print(f"{name:18s} SIGReg = {sigreg_loss(x).item():.5f}")
    # Expected: target ~0; collapsed and scaled large; uniform small-but-nonzero.
