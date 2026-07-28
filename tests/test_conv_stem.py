"""Phase-2 conv-stem tokenizer: a drop-in that must not disturb anything already trained.

Pytest port of `scripts/test_conv_stem.py` (kept as the standalone runnable demo) so the
guarantees run in CI. The claims that matter for the Phase-2 A/B:
  - the conv stem emits the SAME token grid, so pos-embed / predictor / probe are untouched;
  - the LINEAR path is bit-identical and keeps legacy checkpoint keys -- every checkpoint
    trained before Phase 2 must still load;
  - padding is genuinely circular, because CAMELS boxes are periodic and zero/reflect padding
    would inject a spurious edge into exactly the high-k signal the stem exists to preserve.

Real Phase-2 geometry (patch-8, d=1024, 32x32=1024 tokens). Depth is 2 layers only: torch's
CPU build is unstable on the true 24-layer attention, and depth cannot affect tokenizer shape
compatibility, which is all this file claims.
"""
import pytest
import torch

from jepa_loss import JEPA, ConvStem, ViTEncoder, ViTPredictor, random_block_mask

IMG, PATCH, D = 256, 8, 1024
HEADS, LAYERS = 8, 2
GRID = IMG // PATCH          # 32
N = GRID * GRID              # 1024 tokens -- what the conv stem must reproduce
B = 2


def build(stem, seed=0):
    torch.manual_seed(seed)
    return ViTEncoder(img=IMG, patch=PATCH, d=D, heads=HEADS, layers=LAYERS, stem=stem)


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(0)
    return torch.randn(B, 1, IMG, IMG)


def test_both_stems_emit_the_same_token_grid(x):
    with torch.no_grad():
        y_lin, y_conv = build("linear")(x), build("conv")(x)
    assert y_lin.shape == (B, N, D)
    assert y_conv.shape == y_lin.shape, "conv stem must match the linear token layout"


def test_linear_path_is_bit_identical_and_keeps_legacy_keys():
    """Old checkpoints must load unchanged: same seed -> same params, same key namespace."""
    a, b = build("linear").state_dict(), build("linear").state_dict()
    assert list(a) == list(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"linear path not reproducible at {k}"
    assert {k.split(".")[0] for k in a} == {"proj", "pos", "blocks"}


def test_conv_key_namespace_is_disjoint_from_linear():
    """conv_stem.* vs proj.*: a conv checkpoint can never silently half-load into a linear model."""
    top = {k.split(".")[0] for k in build("conv").state_dict()}
    assert "conv_stem" in top and "proj" not in top
    assert not hasattr(build("linear"), "conv_stem"), "linear encoder must not build conv_stem"


def test_conv_stem_is_deterministic(x):
    with torch.no_grad():
        assert torch.equal(build("conv")(x), build("conv")(x))


def test_padding_is_circular(x):
    """Periodic-shift equivariance: rolling the input by one patch must roll the token grid by
    one column. Zero or reflect padding breaks this -- and would fabricate an edge in a
    periodic simulation box."""
    stem = ConvStem(PATCH, D).eval()
    with torch.no_grad():
        t0 = stem(x).view(B, GRID, GRID, D)
        shifted = stem(torch.roll(x, shifts=PATCH, dims=3)).view(B, GRID, GRID, D)
        expected = torch.roll(t0, shifts=1, dims=2)
    err = (shifted - expected).abs().max().item()
    assert err < 1e-4, f"circular equivariance broken (max err {err:.2e}) -- padding not circular?"


def test_conv_stem_rejects_non_power_of_two_patch():
    with pytest.raises(ValueError, match="power-of-2"):
        ConvStem(patch=12, d=64)


def test_encoder_rejects_an_unknown_stem():
    with pytest.raises(ValueError, match="unknown stem"):
        ViTEncoder(img=IMG, patch=PATCH, d=64, heads=4, layers=1, stem="wavelet")


@pytest.mark.parametrize("stem", ["linear", "conv"])
def test_masked_forward_and_full_jepa_step_run_under_both_stems(x, stem):
    enc = build(stem)
    pred = ViTPredictor(N, d=D, pred_d=384, heads=6, layers=6)
    jepa = JEPA(enc, pred, loss_mode="lejepa", var_coef=5.0, cov_coef=4e-2, target_norm=True)
    ctx_idx, tgt_idx = random_block_mask(GRID, block=8, n_blocks=4, device=x.device)

    ctx = enc(x, keep=ctx_idx)
    assert ctx.shape == (B, len(ctx_idx), D), "masked forward must drop exactly the target tokens"

    loss, _ = jepa(x, ctx_idx, tgt_idx)
    assert torch.isfinite(loss).all(), f"{stem}: non-finite loss"
