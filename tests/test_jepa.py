"""JEPA wiring: the three anti-collapse modes and the structural switches between them.

These are cheap tests of expensive mistakes. Each mode is a different answer to "what stops
the representation collapsing", and they differ in what gets built (a frozen EMA teacher or
not), what carries gradient, and what gets logged. Getting that wrong does not raise -- it
produces a training run that looks fine and learns nothing.

Tiny geometry (32px, patch 4, d=64, 1 layer): this file tests wiring, not capacity.
"""
import pytest
import torch

from jepa_loss import JEPA, LOSS_MODES, ViTEncoder, ViTPredictor, random_block_mask

IMG, PATCH, D, GRID = 32, 4, 64, 8
N = GRID * GRID
DEV = torch.device("cpu")


def make(loss_mode="lejepa", **kw):
    enc = ViTEncoder(img=IMG, patch=PATCH, d=D, heads=4, layers=1)
    pred = ViTPredictor(N, d=D, pred_d=32, heads=4, layers=1)
    return JEPA(enc, pred, loss_mode=loss_mode, **kw)


@pytest.fixture
def batch():
    x = torch.randn(4, 1, IMG, IMG)
    ctx_idx, tgt_idx = random_block_mask(GRID, block=2, n_blocks=2, device=DEV)
    return x, ctx_idx, tgt_idx


def test_encoder_token_shape_and_masked_subsetting():
    enc = ViTEncoder(img=IMG, patch=PATCH, d=D, heads=4, layers=1)
    x = torch.randn(2, 1, IMG, IMG)
    assert enc(x).shape == (2, N, D)
    keep = torch.arange(10)
    assert enc(x, keep=keep).shape == (2, 10, D), "keep= must encode only the context tokens"


@pytest.mark.parametrize("mode", list(LOSS_MODES))
def test_every_mode_produces_a_finite_loss_and_trains(batch, mode):
    x, ctx_idx, tgt_idx = batch
    jepa = make(mode, var_coef=5.0, cov_coef=4e-2, target_norm=True)
    loss, tgt = jepa(x, ctx_idx, tgt_idx)
    assert torch.isfinite(loss).all()
    assert tgt.shape[1] == len(tgt_idx)

    loss.backward()
    grads = [p.grad for p in jepa.context_encoder.parameters() if p.grad is not None]
    assert grads, f"{mode}: no gradient reached the encoder"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("mode,teacher", [(m, LOSS_MODES[m]["needs_teacher"]) for m in LOSS_MODES])
def test_teacher_is_built_only_where_the_mode_needs_one(mode, teacher):
    """lejepa/visreg replace the teacher with a distributional regularizer -- so allocating one
    would waste a whole encoder's parameters of dead memory per GPU (it matters at ViT-L)."""
    jepa = make(mode)
    assert (jepa.target_encoder is not None) == teacher


def test_ema_teacher_is_frozen():
    jepa = make("ema")
    assert all(not p.requires_grad for p in jepa.target_encoder.parameters())


def test_step_ema_moves_the_teacher_only_in_ema_mode():
    jepa = make("ema", ema_decay=0.9)
    before = [p.clone() for p in jepa.target_encoder.parameters()]
    with torch.no_grad():
        for p in jepa.context_encoder.parameters():
            p.add_(1.0)                       # force a divergence for the EMA to chase
    jepa.step_ema()
    assert any(not torch.equal(a, b) for a, b in zip(before, jepa.target_encoder.parameters()))

    lejepa = make("lejepa")
    lejepa.step_ema()                          # must be a no-op, not an AttributeError


def test_stop_grad_is_the_switch_that_severs_the_target(batch):
    """The collapse control from the study: with stop_grad the target is detached; without it,
    gradient flows into BOTH sides and a constant vector becomes the global minimum."""
    x, ctx_idx, tgt_idx = batch
    _, tgt_on = make("ema", stop_grad=True)(x, ctx_idx, tgt_idx)
    assert not tgt_on.requires_grad

    _, tgt_off = make("ema", stop_grad=False)(x, ctx_idx, tgt_idx)
    assert tgt_off.requires_grad


def test_collapse_detectors_are_stashed_in_range(batch):
    """The trainer's abort guard reads these every step; they must exist and be sane."""
    x, ctx_idx, tgt_idx = batch
    jepa = make("lejepa", var_coef=5.0, cov_coef=4e-2)
    jepa(x, ctx_idx, tgt_idx)
    assert 1.0 <= jepa.last_eff_rank <= D, f"eff_rank {jepa.last_eff_rank} outside [1, {D}]"
    assert jepa.last_tgt_std > 0
    for name in ("last_pred", "last_reg", "last_var", "last_cov"):
        assert torch.isfinite(torch.tensor(getattr(jepa, name)))


def test_unknown_loss_mode_is_rejected_at_construction():
    """Fail at build time, not 4000 steps into a pod run."""
    with pytest.raises(ValueError, match="unknown loss_mode"):
        make("contrastive")
