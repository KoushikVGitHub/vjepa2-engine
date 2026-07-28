"""The anisotropic-collapse result, as an executable claim.

This is the repo's central finding (study/notes/collapse_resolution.md): SIGReg's Cramer-Wold
sketch tests each 1-D projection MARGINALLY, so a low-rank blob whose marginals still look
like N(0,1) sails through it with a vanishing gradient -- while the VICReg off-diagonal
covariance term sees it immediately. That is why `--var-coef/--cov-coef` exist, and why
effective rank (not target-std) is the collapse detector that matters.

The README quotes ||grad sigreg|| ~ 2e-4 vs ~1.25 for the covariance penalty on the real
ViT-L shape. `test_sigreg_is_blind_where_covariance_is_not` rebuilds that comparison at CI
scale so the claim is re-checked on every push instead of resting on a screenshot.
"""
import torch
import torch.nn.functional as F

from sigreg import sigreg_loss, variance_covariance_reg


def rank_r_batch(n=4096, d=64, r=2):
    """A rank-r embedding whose MARGINALS are exactly N(0,1).

    Rows of A are unit-norm, so z_j = <A_j, u> has variance 1 for every dim j: per-dim std
    looks perfectly healthy, and target-std -- the naive collapse detector -- sees nothing
    wrong. But the batch lives in an r-dim subspace of d, which is precisely the dimensional
    collapse that silently caps probe R^2.
    """
    A = F.normalize(torch.randn(d, r), dim=1)
    return torch.randn(n, r) @ A.t()


def effective_rank(z):
    """Participation ratio PR = tr(C)^2 / ||C||_F^2, in [1, d] -- the trainer's `eff_rank`."""
    z = z - z.mean(dim=0, keepdim=True)
    C = (z.t() @ z) / z.size(0)
    return (torch.diagonal(C).sum() ** 2 / C.pow(2).sum()).item()


def test_effective_rank_recovers_the_true_subspace_dimension():
    """The detector itself must be correct before anything it reports can be trusted."""
    for r in (1, 2, 8):
        assert abs(effective_rank(rank_r_batch(r=r)) - r) < 0.25 * r
    # ...and a genuinely isotropic batch should read out near full rank.
    assert effective_rank(torch.randn(8192, 64)) > 55


def test_collapsed_batch_has_healthy_per_dimension_std():
    """The trap, stated as a test: std is ~1.0 while the batch is rank-2 of 64.

    Anyone monitoring only tgt_std would call this run healthy.
    """
    z = rank_r_batch()
    assert abs(z.std(dim=0).mean().item() - 1.0) < 0.05
    assert effective_rank(z) < 3


def test_sigreg_is_blind_where_covariance_is_not():
    """The load-bearing claim: at a dimensionally-collapsed point the covariance penalty's
    gradient is ORDERS of magnitude stronger than SIGReg's, so it is the term that actually
    escapes the low-rank basin."""
    z = rank_r_batch()

    za = z.clone().requires_grad_()
    sigreg_loss(za, generator=torch.Generator().manual_seed(3)).backward()

    zb = z.clone().requires_grad_()
    _, cov = variance_covariance_reg(zb)
    cov.backward()

    g_sigreg, g_cov = za.grad.norm().item(), zb.grad.norm().item()
    assert g_sigreg < 1e-2, f"SIGReg gradient unexpectedly large at collapse: {g_sigreg:.2e}"
    assert g_cov > 100 * g_sigreg, (
        f"covariance penalty lost its advantage: |grad cov| {g_cov:.3e} "
        f"vs |grad sigreg| {g_sigreg:.3e} (expected >100x)"
    )


def test_covariance_term_separates_decorrelated_from_low_rank():
    """cov is the RANK knob: ~0 for an isotropic batch, large for a collapsed one."""
    _, cov_ok = variance_covariance_reg(torch.randn(4096, 64))
    _, cov_bad = variance_covariance_reg(rank_r_batch())
    assert cov_ok.item() < 0.05
    assert cov_bad.item() > 10 * cov_ok.item()


def test_variance_hinge_is_the_scale_knob():
    """var is the SCALE knob: silent at std >= gamma, active below it, and its gradient does
    NOT vanish as std -> 0 (which is exactly what SIGReg's does)."""
    # Comfortably above the hinge. (At std == gamma exactly the hinge is not silent: finite-batch
    # std fluctuates either side of 1.0, so a real N(0,1) sample still scores ~5e-3. That is the
    # term working as specified, not a defect -- so the "silent" claim is tested with margin.)
    healthy = torch.randn(2048, 32) * 1.5
    var_ok, _ = variance_covariance_reg(healthy, gamma=1.0)
    assert var_ok.item() == 0.0

    shrunk = torch.randn(2048, 32) * 0.01
    var_bad, _ = variance_covariance_reg(shrunk, gamma=1.0)
    assert var_bad.item() > 0.9, "hinge should read ~gamma - std when the batch is shrinking"

    z = (torch.randn(2048, 32) * 1e-3).requires_grad_()
    variance_covariance_reg(z, gamma=1.0)[0].backward()
    assert z.grad.norm().item() > 0, "variance hinge must keep pushing on a near-dead direction"


def test_regularizers_are_finite_on_a_degenerate_batch():
    """Guard the eps floors: an exactly-constant batch must not produce NaN/Inf."""
    z = torch.full((512, 32), 0.7)
    var_l, cov_l = variance_covariance_reg(z)
    assert torch.isfinite(var_l) and torch.isfinite(cov_l)
