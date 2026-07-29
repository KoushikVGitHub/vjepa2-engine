"""CPU test for the S3 control: the 'convdisjoint' stem's CONVOLUTIONAL receptive field is exactly
one patch (no cross-patch overlap), while 'conv' bleeds across patch boundaries. Proven by
perturbing a single interior patch and checking which output tokens change.

Why strip GroupNorm for the geometry proof: GroupNorm(1,C) normalizes over ALL spatial positions,
so on the real module any perturbation shifts the global norm stats and nudges every token. That
global coupling is SHARED by conv and convdisjoint (held constant across the S3 contrast), so it is
not the variable under test. To isolate the convolution GEOMETRY we swap GroupNorm->Identity here,
then separately confirm both real modules are shape-compatible drop-ins through the encoder.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn as nn
from jepa_loss import ConvStem, ViTEncoder

IMG, PATCH, D = 64, 8, 64          # grid = 8x8 = 64 tokens; small + fast on CPU
GRID = IMG // PATCH


def strip_norm(stem):
    "Replace GroupNorm with Identity so the test sees ONLY the conv receptive-field geometry."
    for i, m in enumerate(stem.net):
        if isinstance(m, nn.GroupNorm):
            stem.net[i] = nn.Identity()
    return stem


@torch.no_grad()
def changed_tokens(stem, x, r, c):
    "Set of output-token (row,col) indices that change when input patch (r,c) is perturbed."
    stem.eval()
    t0 = stem(x)
    x2 = x.clone()
    x2[:, :, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] += 3.0   # perturb ONE whole patch
    t1 = stem(x2)
    diff = (t1 - t0).abs().flatten(0, 1).sum(-1)            # (GRID*GRID,) per-token L1 change
    idx = (diff > 1e-5).nonzero().flatten().tolist()
    return {(i // GRID, i % GRID) for i in idx}


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 1, IMG, IMG)
    r, c = 3, 4                                             # an INTERIOR patch (no wrap-around)

    dis = strip_norm(ConvStem(PATCH, D, overlap=False))
    ov = strip_norm(ConvStem(PATCH, D, overlap=True))
    dch = changed_tokens(dis, x, r, c)
    och = changed_tokens(ov, x, r, c)

    # convdisjoint: perturbing patch (r,c) changes EXACTLY token (r,c) -> disjoint receptive field
    assert dch == {(r, c)}, f"convdisjoint NOT disjoint: patch ({r},{c}) changed tokens {sorted(dch)}"
    print(f"[ok] convdisjoint: perturbing patch ({r},{c}) changes exactly 1 token {(r, c)} (disjoint RF)")

    # conv (overlap): the SAME perturbation also changes neighbors -> overlapping receptive field
    assert (r, c) in och and len(och) > 1, f"conv did not overlap: changed {sorted(och)}"
    print(f"[ok] conv (overlap): same perturbation changes {len(och)} tokens incl neighbors "
          f"{sorted(och - {(r, c)})} (overlapping RF)")

    # shape-compat: both are drop-in stems emitting (1, GRID*GRID, D) at the tokenizer stage
    for name in ("conv", "convdisjoint"):
        enc = ViTEncoder(img=IMG, patch=PATCH, d=D, heads=4, layers=2, stem=name)
        enc.probe_stage = "tokenizer"
        enc.eval()
        with torch.no_grad():
            tok = enc(x)
        assert tok.shape == (1, GRID * GRID, D), (name, tok.shape)
        print(f"[ok] {name}: encoder tokenizer emits {tuple(tok.shape)} (drop-in)")

    # param counts: S3 is an ARCHITECTURE/overlap control, NOT param-matched -- report honestly.
    pc = lambda o: sum(p.numel() for p in ConvStem(PATCH, D, overlap=o).parameters())
    print(f"[info] params: conv(overlap)={pc(True) / 1e3:.1f}K  convdisjoint={pc(False) / 1e3:.1f}K "
          f"(convdisjoint has fewer params, so a NULL result conv~=convdisjoint is conservative; "
          f"the param-matched disjoint control is H1's mlp).")

    print("\nALL GREEN -- convdisjoint RF is exactly one patch; conv RF overlaps across patches.")


if __name__ == "__main__":
    main()
