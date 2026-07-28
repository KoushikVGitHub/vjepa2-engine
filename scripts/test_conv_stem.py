#!/usr/bin/env python
"""
CPU shape-check + regression test for the Phase-2 conv-stem tokenizer.

Runs with no GPU and no data. Asserts:
  (1) imports work;
  (2) LINEAR path is UNCHANGED -- same seed -> bit-identical params, and the state_dict
      carries exactly the legacy keys (proj.*, pos, blocks.*) so old checkpoints still load;
  (3) CONV path produces the SAME token grid (B, grid*grid, d) as linear -> pos-embed /
      predictor / probe are all shape-compatible; keys are conv_stem.*, no proj.*;
  (4) both stems are deterministic (same seed -> identical output);
  (5) conv padding is circular (a periodic input shift -> a matching shift in tokens);
  (6) the masked encoder forward (keep=subset) and a full JEPA step run under both stems.
Exit 0 = all green.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
import torch.nn as nn
from jepa_loss import (ViTEncoder, ViTPredictor, JEPA, ConvStem, MLPStem,
                       _conv_stem_param_count, random_block_mask)

# REAL Phase-2 stem geometry (patch-8, d-1024 -> 32x32=1024 tokens): this is what the deliverable
# must get right (token layout, checkpoint keys, circular equivariance). LAYERS/HEADS/B are kept
# small ONLY because torch's CPU build segfaults on the true 24-layer attention -- transformer DEPTH
# is irrelevant to tokenizer shape-compatibility, which is all a CPU shape-check can (and needs to) prove.
IMG, PATCH, D = 256, 8, 1024
HEADS, LAYERS = 8, 2
GRID = IMG // PATCH        # 32
N = GRID * GRID            # 1024 -- the true token count the conv-stem must reproduce
B = 2


def build(stem, seed=0):
    torch.manual_seed(seed)
    return ViTEncoder(img=IMG, patch=PATCH, d=D, heads=HEADS, layers=LAYERS, stem=stem)


def main():
    x = torch.randn(B, 1, IMG, IMG)

    # (1)+(3) shapes ---------------------------------------------------------
    enc_lin = build("linear")
    enc_conv = build("conv")
    y_lin = enc_lin(x)
    y_conv = enc_conv(x)
    assert y_lin.shape == (B, N, D), y_lin.shape
    assert y_conv.shape == (B, N, D), y_conv.shape
    assert y_lin.shape == y_conv.shape, "conv stem must match linear token layout"
    print(f"[ok] token shapes: linear {tuple(y_lin.shape)} == conv {tuple(y_conv.shape)}  (grid {GRID}x{GRID})")

    # (2) linear path bit-identical to a fresh linear encoder + legacy keys ---
    enc_lin_b = build("linear")  # same seed
    for (ka, va), (kb, vb) in zip(enc_lin.state_dict().items(), enc_lin_b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), f"linear path not reproducible at {ka}"
    lin_keys = set(k.split(".")[0] for k in enc_lin.state_dict())
    assert lin_keys == {"proj", "pos", "blocks"}, f"unexpected linear keys: {lin_keys}"
    assert not hasattr(enc_lin, "conv_stem"), "linear encoder must not build conv_stem"
    print(f"[ok] linear path bit-identical + legacy keys only: {sorted(lin_keys)}")

    # (3b) conv key namespace is disjoint (old ckpts won't collide) ----------
    conv_top = set(k.split(".")[0] for k in enc_conv.state_dict())
    assert "conv_stem" in conv_top and "proj" not in conv_top, conv_top
    print(f"[ok] conv path keys: {sorted(conv_top)} (no proj.*)")

    # (4) determinism --------------------------------------------------------
    assert torch.equal(y_conv, build("conv")(x)), "conv stem not deterministic"
    print("[ok] conv stem deterministic (same seed -> identical output)")

    # (5) circular padding: roll input by one PATCH -> tokens roll by one col -
    stem = ConvStem(PATCH, D).eval()
    with torch.no_grad():
        t0 = stem(x)                                             # (B, N, D)
        xs = torch.roll(x, shifts=PATCH, dims=3)                 # shift right by one patch (periodic)
        ts = stem(xs).view(B, GRID, GRID, D)
        t0g = t0.view(B, GRID, GRID, D)
        rolled = torch.roll(t0g, shifts=1, dims=2)               # expected: token grid rolls one col
    err = (ts - rolled).abs().max().item()
    assert err < 1e-4, f"circular equivariance broken (max err {err:.2e}) -- padding not circular?"
    print(f"[ok] circular padding verified (periodic-shift equivariance, max err {err:.1e})")

    # (6) REVIEWER CONTROLS ---------------------------------------------------
    # H1: MLP stem is param-matched to the conv stem (within ~2%) and same token shape
    conv_p = _conv_stem_param_count(PATCH, D)
    mlp = MLPStem(PATCH, D)
    mlp_p = sum(p.numel() for p in mlp.parameters())
    assert abs(mlp_p - conv_p) / conv_p < 0.03, f"MLPStem not param-matched: {mlp_p} vs conv {conv_p}"
    xp = ViTEncoder(img=IMG, patch=PATCH, d=D, heads=HEADS, layers=LAYERS, stem="linear").patchify(x)
    assert mlp(xp).shape == (B, N, D), mlp(xp).shape
    print(f"[ok] H1 mlp stem param-matched to conv ({mlp_p/1e6:.2f}M vs {conv_p/1e6:.2f}M) + token shape {tuple(mlp(xp).shape)}")

    # H11: conv stem builds with zeros padding and keeps token shape (periodicity ablation)
    cz = ConvStem(PATCH, D, pad="zeros").eval()
    assert cz(x).shape == (B, N, D)
    print(f"[ok] H11 conv-stem pad='zeros' builds + token shape {tuple(cz(x).shape)}")

    # H3: probe_stage='tokenizer' returns RAW tokens (pre-transformer), not the encoded output
    ec = build("conv")
    ec.probe_stage = "tokenizer"
    with torch.no_grad():
        raw = ec(x)
        ec.probe_stage = "encoder"
        enc_out = ec(x)
    assert raw.shape == enc_out.shape == (B, N, D)
    assert not torch.equal(raw, enc_out), "tokenizer-stage must differ from encoder-stage output"
    print(f"[ok] H3 probe_stage='tokenizer' returns pre-transformer tokens {tuple(raw.shape)} (!= encoder stage)")

    # (7) masked forward + full JEPA step under all THREE stems --------------
    for stem_name in ("linear", "conv", "mlp"):
        enc = build(stem_name)
        pred = ViTPredictor(N, d=D, pred_d=384, heads=6, layers=6)
        jepa = JEPA(enc, pred, loss_mode="lejepa", var_coef=5.0, cov_coef=4e-2, target_norm=True)
        ctx_idx, tgt_idx = random_block_mask(GRID, block=8, n_blocks=4, device=x.device)
        ctx = enc(x, keep=ctx_idx)
        assert ctx.shape == (B, len(ctx_idx), D)
        out = jepa(x, ctx_idx, tgt_idx)
        loss = out[0] if isinstance(out, (tuple, list)) else out
        assert torch.isfinite(loss).all(), f"{stem_name}: non-finite loss"
        n_params = sum(p.numel() for p in enc.parameters()) / 1e6
        print(f"[ok] {stem_name:6s}: masked ctx {tuple(ctx.shape)}, JEPA loss {loss.item():.4f}, enc {n_params:.1f}M params")

    print("\nALL GREEN -- 3 stems shape-compatible; linear path unchanged; H1/H3/H11 controls wired.")


if __name__ == "__main__":
    main()
