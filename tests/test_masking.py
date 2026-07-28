"""I-JEPA block masking.

The context/target split is the supervision signal itself. A silent bug here (overlap, an
empty context, a target ratio far off the intended ~15-25%) would not crash -- it would just
make the prediction task trivially easy, which is exactly the low-rank cheat that feeds
dimensional collapse. Phase 1 swept this ratio, so its structure is worth pinning down.
"""
import torch

from jepa_loss import random_block_mask

GRID, BLOCK = 32, 8
DEV = torch.device("cpu")


def test_context_and_target_partition_the_grid():
    """Disjoint and exhaustive: every token is either seen or predicted, never both."""
    ctx, tgt = random_block_mask(GRID, BLOCK, DEV, n_blocks=4)
    assert len(set(ctx.tolist()) & set(tgt.tolist())) == 0
    assert sorted(ctx.tolist() + tgt.tolist()) == list(range(GRID * GRID))


def test_target_ratio_respects_the_block_budget():
    """Union of n blocks, so the ratio is AT MOST n*block^2/grid^2 -- less when they overlap."""
    for n_blocks in (1, 4, 8):
        _, tgt = random_block_mask(GRID, BLOCK, DEV, n_blocks=n_blocks)
        ratio = len(tgt) / (GRID * GRID)
        assert 0 < ratio <= n_blocks * BLOCK ** 2 / GRID ** 2 + 1e-9
        assert ratio >= BLOCK ** 2 / GRID ** 2 - 1e-9, "at least one full block must be masked"


def test_a_single_block_is_contiguous_and_square():
    """Masked regions are spatial blocks, not scattered tokens -- the whole point of I-JEPA."""
    _, tgt = random_block_mask(GRID, BLOCK, DEV, n_blocks=1)
    assert len(tgt) == BLOCK ** 2
    rows, cols = tgt // GRID, tgt % GRID
    assert rows.max() - rows.min() == BLOCK - 1
    assert cols.max() - cols.min() == BLOCK - 1


def test_context_is_never_empty_at_production_settings():
    """The encoder needs something to read: n-blocks=4, block=8 on a 32x32 grid (the trained
    default) must always leave context tokens behind."""
    for _ in range(25):
        ctx, _ = random_block_mask(GRID, BLOCK, DEV, n_blocks=4)
        assert len(ctx) > 0


def test_mask_is_random_across_calls():
    """A fixed mask would let the encoder memorize one hole instead of learning structure."""
    masks = {tuple(random_block_mask(GRID, BLOCK, DEV, n_blocks=2)[1].tolist()) for _ in range(8)}
    assert len(masks) > 1
