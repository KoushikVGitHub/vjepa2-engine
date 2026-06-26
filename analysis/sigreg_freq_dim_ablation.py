"""Day 2 ablation - SIGReg's detection power vs (a) CF frequency range, (b) dimension.

Motivation: the sigreg.py sanity demo flagged `uniform` (zero-mean, ~unit-var, NOT
Gaussian) as ~0 at D=64 -- i.e. SIGReg did NOT detect it. This file diagnoses WHY,
and turns the surprise into a documented limitation of projection-based normality tests.

Two experiments:
  1) freq_scale sweep: does pushing CF frequencies higher recover the `uniform` signal?
     -> NO. Variance errors are detected at LOW freq (CFs decay to 0 at high t,
        Riemann-Lebesgue), and uniform stays invisible regardless of freq. Hypothesis
        "we under-sample high frequencies" is FALSIFIED.
  2) dimension sweep: is `uniform` detectable at low D and invisible at high D?
     -> YES. At D=1 uniform scores ~160x the Gaussian control; by D=64 it's ~1x.
        Random 1-D projections of high-dim data are ~Gaussian by CLT (Diaconis-Freedman),
        so per-axis non-Gaussianity that lives only in the joint is invisible to any
        Cramer-Wold / 1-D-marginal test.

Takeaway (for the writeup): SIGReg reliably catches COLLAPSE and ANISOTROPY (wrong
variance / low-rank directions show up in projections), but is BLIND to non-Gaussian
structure that survives only in the high-dim joint -- because the quantity it tests
(1-D marginals) is exactly what projection Gaussianizes. "Passes SIGReg" != "latent is
Gaussian in every sense"; it means "anti-collapse + isotropy satisfied", which is the
property the objective actually needs.
"""
import torch
import torch.nn.functional as F


def random_directions(dim, n_proj, device, generator=None):
    v = torch.randn(dim, n_proj, device=device, generator=generator)
    return F.normalize(v, dim=0)


def sigreg_loss(z, n_proj=256, n_freq=64, freq_scale=1.0):
    """SIGReg regularizer with an exposed `freq_scale` dial (1.0 == sigreg.py default)."""
    B, D = z.shape
    V = random_directions(D, n_proj, z.device)
    P = z @ V
    t = torch.randn(n_freq, device=z.device) * freq_scale
    tp = P.unsqueeze(-1) * t
    cos = tp.cos().mean(dim=0)
    sin = tp.sin().mean(dim=0)
    target = torch.exp(-0.5 * t ** 2)
    return ((cos - target) ** 2 + sin ** 2).mean()


def experiment_frequency():
    print("=== (1) frequency sweep @ D=64 (hypothesis: higher freq recovers uniform) ===")
    torch.manual_seed(0)
    B, D = 4096, 64
    cases = {
        "N(0,I) target": torch.randn(B, D),
        "collapsed":     torch.zeros(B, D) + 0.01 * torch.randn(1, D),
        "scaled var=25": 5.0 * torch.randn(B, D),
        "uniform":       (torch.rand(B, D) - 0.5) * 3.46,
    }
    scales = [1.0, 2.0, 3.0, 5.0]
    print(f"{'case':16s} " + " ".join(f"x{s:<8}" for s in scales))
    for name, x in cases.items():
        row = []
        for s in scales:
            torch.manual_seed(0)
            row.append(f"{sigreg_loss(x, freq_scale=s).item():.5f}")
        print(f"{name:16s} " + " ".join(f"{v:<9}" for v in row))
    print("-> uniform flat; scaled FALLS with freq. Frequency hypothesis FALSIFIED.\n")


def experiment_dimension():
    print("=== (2) dimension sweep (real cause: CLT/Diaconis-Freedman) ===")
    B = 8192
    print(f"{'D':>4}  {'uniform':>10}  {'gaussian(ctrl)':>14}  ratio")
    for D in [1, 2, 4, 8, 16, 64]:
        torch.manual_seed(0)
        u = (torch.rand(B, D) - 0.5) * 3.46
        torch.manual_seed(0)
        gg = torch.randn(B, D)
        torch.manual_seed(1); lu = sigreg_loss(u)
        torch.manual_seed(1); lg = sigreg_loss(gg)
        print(f"{D:>4}  {lu.item():>10.5f}  {lg.item():>14.5f}  {lu.item()/max(lg.item(),1e-9):>5.1f}x")
    print("-> uniform glaring at D=1 (~160x), invisible by D=64 (~1x). Diagnosis CONFIRMED.")


if __name__ == "__main__":
    experiment_frequency()
    experiment_dimension()
