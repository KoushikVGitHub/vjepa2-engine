"""Phase-3: WHERE in the 24 blocks does the cosmology go?

The tokenizer-beats-encoder result has a standard alternative explanation that must be ruled out
before recommending anything: in ViTs the best linear-probe features often sit at INTERMEDIATE
depth, and only the last blocks specialise to the pretext. If layer ~12 recovers most of the
tokenizer's skill, the fault is "the top of the stack is pretext-specialised" (fix: read out
mid-stack) rather than "the transformer destroys the signal" (fix: retrain / far fewer layers).
Those imply different next steps, so measure it.

One frozen pass per suite yields every depth at once: run the stem, then walk blocks.layers,
pooling (mean+std over tokens) after each requested depth. Depth 24 is asserted against the
encoder's own forward() so the walk is provably the same computation.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from probe import load_frozen_encoder, sim_split                       # noqa: E402
from phase3_feats import make_dataset                                  # noqa: E402
from phase3_diag import standardizer, ridge_fit, predict, metrics, pick_alpha, PARAMS  # noqa: E402


@torch.no_grad()
def featurize_depths(enc, ds, device, depths, bs, workers):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    drop_last=False, pin_memory=True)
    acc = {d: [] for d in depths}
    ys = []
    layers = enc.blocks.layers
    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            h = enc._tokens(x)
            if 0 in acc:
                acc[0].append(pool(h))
            for li, blk in enumerate(layers, start=1):
                h = blk(h)
                if li in acc:
                    acc[li].append(pool(h))
        ys.append(y.numpy().astype(np.float32))
        if i == 0:
            # Provenance check on the first batch: the manual walk must BE encoder.forward().
            # Compared under the same autocast (an fp32 reference vs a bf16 walk fails on
            # precision alone, which is not the thing being checked) and in relative terms.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                ref = enc(x)
            rel = ((ref.float() - h.float()).norm() / ref.float().norm()).item()
            print(f"  [check] ||forward - block-walk|| / ||forward|| = {rel:.2e}", flush=True)
            assert rel < 1e-2, f"manual block walk != encoder.forward() (rel err {rel:.3e})"
        if i % 40 == 0:
            print(f"  {i * bs}/{len(ds)}", flush=True)
    return {d: np.concatenate(v) for d, v in acc.items()}, np.concatenate(ys)


def pool(h):
    h = h.float()
    return torch.cat([h.mean(1), h.std(1)], dim=1).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/checkpoints/ckpt_p2c_conv_s0.pt")
    ap.add_argument("--data-root", default="/dev/shm/cdata")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 2, 4, 8, 12, 16, 20, 24])
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--random-init", action="store_true",
                    help="attribution control: same sweep on UNTRAINED weights, so the shallow-depth "
                         "score can be split into 'what mixing does' vs 'what pretraining taught'")
    args = ap.parse_args()

    torch.set_num_threads(8)
    enc = load_frozen_encoder(args.ckpt, "cuda", random_init=args.random_init, img=256, patch=8,
                              d=1024, heads=16, layers=24, stem="conv", stem_pad="circular")
    print(f"[depth] {'RANDOM-INIT' if args.random_init else 'trained'} encoder", flush=True)
    feats, ys = {}, {}
    for suite in ["IllustrisTNG", "SIMBA"]:
        ds = make_dataset(args.data_root, "Mgas", suite)
        print(f"[depth] featurizing {suite}", flush=True)
        feats[suite], ys[suite] = featurize_depths(enc, ds, "cuda", args.depths,
                                                   args.batch, args.workers)

    tr, va, te = sim_split(15000)
    ttr, tva, tte = sim_split(15000)
    Ys, Yt = ys["IllustrisTNG"][:, :2].astype(np.float64), ys["SIMBA"][:, :2].astype(np.float64)
    print(f"\n{'depth':>6s} | {'in-suite ITNG':>16s} | {'x-suite SIMBA selfnorm':>24s} "
          f"| {'x-suite r2aff':>15s} | {'SIMBA oracle':>16s}")
    print("-" * 92)
    for d in args.depths:
        Xs = feats["IllustrisTNG"][d].astype(np.float64)
        Xt = feats["SIMBA"][d].astype(np.float64)
        mu, sd = standardizer(Xs[tr])
        a, _ = pick_alpha((Xs[tr] - mu) / sd, Ys[tr], (Xs[va] - mu) / sd, Ys[va])
        W, yb = ridge_fit((Xs[tr] - mu) / sd, Ys[tr], a)
        mi = metrics(Ys[te], predict((Xs[te] - mu) / sd, W, yb))
        mut, sdt = standardizer(Xt)
        mx = metrics(Yt, predict((Xt - mut) / sdt, W, yb))
        mo, so = standardizer(Xt[ttr])
        ao, _ = pick_alpha((Xt[ttr] - mo) / so, Yt[ttr], (Xt[tva] - mo) / so, Yt[tva])
        Wo, ybo = ridge_fit((Xt[ttr] - mo) / so, Yt[ttr], ao)
        mor = metrics(Yt[tte], predict((Xt[tte] - mo) / so, Wo, ybo))
        f = lambda m, k="r2": f"{m[PARAMS[0]][k]:+.3f}/{m[PARAMS[1]][k]:+.3f}"
        print(f"{d:>6d} | {f(mi):>16s} | {f(mx):>24s} | "
              f"{f(mx, 'r2_affine'):>15s} | {f(mor):>16s}", flush=True)


if __name__ == "__main__":
    main()
