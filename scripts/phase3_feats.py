"""Phase-3 diagnosis: dump POOLED frozen features once per (encoder, stage, suite).

Why this exists: every Phase-3 readout question (ridge-head fairness, tokenizer-stage
transfer, feature-shift geometry, few-shot adaptation) needs the SAME frozen features and
differs only in what is fitted on top. Re-running run_probe.py per question costs a full
ViT-L pass over 15k maps each time. Dump once -> every downstream experiment is CPU-seconds.

Pooling is FIXED (mean + std over tokens), deliberately: the deployed ProbeHead uses a
LEARNED attentive pool, which is itself trainable capacity that can overfit the source suite.
A fixed pool is the honest analog of the pk baseline's fixed feature vector, so a ridge on
these features is an apples-to-apples fairness control (Phase-3 experiment B).

Matches run_probe.py exactly on: dataset construction (log10, min_std=0.05, cached manifest
=> per-suite self-normalised inputs), encoder config, and probe_stage.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.fields import FieldMapDataset          # noqa: E402
from probe import load_frozen_encoder            # noqa: E402


def make_dataset(data_root, field, suite):
    npy = os.path.join(data_root, f"Maps_{field}_{suite}_LH_z=0.00.npy")
    params = os.path.join(data_root, f"params_LH_{suite}.txt")
    return FieldMapDataset(npy, name=field, transform="log10", min_std=0.05,
                           params_path=params, return_params=True, use_cache=True)


@torch.no_grad()
def featurize(enc, ds, device, bs, workers, limit=0):
    if limit:
        ds = Subset(ds, list(range(min(limit, len(ds)))))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    drop_last=False, pin_memory=True)
    xs, ys = [], []
    use_cuda = torch.device(device).type == "cuda"
    for i, (x, y) in enumerate(dl):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            t = enc(x.to(device))
        t = t.float()                                  # (B, T, d)
        pooled = torch.cat([t.mean(dim=1), t.std(dim=1)], dim=1)   # (B, 2d)
        xs.append(pooled.cpu().numpy().astype(np.float32))
        ys.append(y.numpy().astype(np.float32))
        if i % 20 == 0:
            print(f"  featurized {i * bs}/{len(ds)}", flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="/dev/shm/cdata")
    ap.add_argument("--field", default="Mgas")
    ap.add_argument("--suites", nargs="+", default=["IllustrisTNG", "SIMBA"])
    ap.add_argument("--stage", choices=["encoder", "tokenizer"], default="encoder")
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--enc-d", type=int, default=1024)
    ap.add_argument("--enc-heads", type=int, default=16)
    ap.add_argument("--enc-layers", type=int, default=24)
    ap.add_argument("--stem", default="conv")
    ap.add_argument("--stem-pad", default="circular")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", required=True, help="output name prefix, e.g. conv_s0")
    ap.add_argument("--out-dir", default="/dev/shm/feat")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap #maps (0 = all)")
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_cfg = dict(img=args.img, patch=args.patch, d=args.enc_d, heads=args.enc_heads,
                   layers=args.enc_layers, stem=args.stem, stem_pad=args.stem_pad)
    enc = load_frozen_encoder(args.ckpt, device, random_init=args.random_init, **enc_cfg)
    enc.probe_stage = args.stage
    print(f"[feats] {args.tag} stage={args.stage} cfg={enc_cfg}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for suite in args.suites:
        ds = make_dataset(args.data_root, args.field, suite)
        print(f"[feats] {suite}: {len(ds)} maps", flush=True)
        X, Y = featurize(enc, ds, device, args.batch, args.workers, args.limit)
        suffix = "_smoke" if args.limit else ""
        out = os.path.join(args.out_dir, f"{args.tag}_{args.stage}_{suite}{suffix}.npz")
        np.savez(out, X=X, Y=Y)
        print(f"[feats] wrote {out}  X{X.shape} Y{Y.shape}", flush=True)


if __name__ == "__main__":
    main()
