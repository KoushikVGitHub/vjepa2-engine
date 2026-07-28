"""SIGReg (Cramer-Wold / characteristic-function) objective.

The contract: sigreg_loss(z) is ~0 iff z is isotropic Gaussian, and grows as z departs from
it. If this breaks, LeJEPA training silently loses its only anti-collapse signal, so it is
gated here rather than discovered 4000 steps into a pod run.
"""
import pytest
import torch

from sigreg import random_directions, sigreg_loss


def test_random_directions_are_unit_vectors():
    V = random_directions(dim=64, n_proj=32, device=torch.device("cpu"))
    assert V.shape == (64, 32)
    norms = V.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_near_zero_for_isotropic_gaussian(gen):
    """z ~ N(0, I) is the objective's global minimum: every 1-D projection is N(0,1)."""
    z = torch.randn(4096, 64)
    loss = sigreg_loss(z, generator=gen)
    assert loss.item() < 1e-3, f"N(0,I) should score ~0, got {loss.item():.2e}"


def test_large_for_complete_collapse(gen):
    """Every sample mapped to the same vector -- the failure SIGReg exists to catch."""
    z = torch.ones(4096, 64) * 0.3
    loss = sigreg_loss(z, generator=gen)
    assert loss.item() > 0.1, f"collapsed batch should score high, got {loss.item():.2e}"


def test_penalizes_wrong_scale(gen):
    """Correct shape, wrong variance still fails: the target is N(0,1), not 'any Gaussian'."""
    ok = sigreg_loss(torch.randn(4096, 64), generator=gen).item()
    scaled = sigreg_loss(torch.randn(4096, 64) * 5.0, generator=gen).item()
    assert scaled > ok * 10, f"mis-scaled batch scored {scaled:.2e} vs isotropic {ok:.2e}"


def test_deterministic_under_a_seeded_generator():
    """Distributed correctness DEPENDS on this: ranks must draw identical V and t."""
    z = torch.randn(512, 32)
    a = sigreg_loss(z, generator=torch.Generator().manual_seed(7))
    b = sigreg_loss(z, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_differs_across_generator_seeds():
    """Sanity on the above: the sketch really is random, not a constant projection."""
    z = torch.randn(512, 32)
    a = sigreg_loss(z, generator=torch.Generator().manual_seed(7))
    b = sigreg_loss(z, generator=torch.Generator().manual_seed(8))
    assert not torch.equal(a, b)


def test_gradient_flows_to_embeddings(gen):
    z = torch.randn(256, 32, requires_grad=True)
    sigreg_loss(z, generator=gen).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_distributed_mode_requires_a_synced_generator():
    """Without a rank-synced generator each rank would project onto DIFFERENT directions and
    the all-reduced statistic would be meaningless. Fail loudly instead."""
    with pytest.raises(AssertionError):
        sigreg_loss(torch.randn(64, 16), generator=None, distributed=True)
