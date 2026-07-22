"""Dump a checkpoint's REAL rank structure for the mechanism figure (and as a standalone check).

Loads a frozen encoder, encodes one batch, and prints a COMPACT, copy-pasteable summary:
  - effective rank (participation ratio) on the encoder tokens  -- cross-checks the training log
  - the top-N normalized eigenvalue spectrum                    -- the real scree plot for the artifact

Reads the encoder config from the checkpoint's saved `args` (no hardcoded ViT-L dims), so it works
on any run -- cosmology (camels) or the natural-image validation (stl10/cifar10).

  python scripts/spectrum_dump.py --ckpt /workspace/ckpt_visreg_cov.pt --dataset camels --field Mgas
  python scripts/spectrum_dump.py --ckpt /workspace/ckpt_stl_visreg.pt --dataset stl10 --data-root /workspace/img_data
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from probe import load_frozen_encoder


def get_batch(args, device):
    if args.dataset == "camels":
        from data.fields import FieldMapDataset
        npy = os.path.join(args.data_root, f"Maps_{args.field}_{args.suite}_LH_z=0.00.npy")
        ds = FieldMapDataset(npy, name=args.field, transform="log10", min_std=0.05, augment=False)
    else:
        from data.images import GrayImageDataset
        ds = GrayImageDataset(args.data_root, name=args.dataset, img=args.img, augment=False)
    idx = torch.randperm(len(ds))[:args.batch]
    return torch.stack([ds[int(i)] for i in idx]).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="camels", choices=["camels", "stl10", "cifar10"])
    ap.add_argument("--field", default="Mgas")
    ap.add_argument("--suite", default="IllustrisTNG")
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--img", type=int, default=96, help="only used to build the image dataset if the "
                                                        "checkpoint args lack it")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--topn", type=int, default=40, help="eigenvalues to report for the scree plot")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encoder config from the checkpoint's saved args -> no hardcoded dims.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    enc_kw = dict(img=a.get("img", 256), patch=a.get("patch", 16),
                  d=a.get("d", 1024), heads=a.get("heads", 16), layers=a.get("layers", 24))
    if a.get("img"):                      # keep the image dataset resolution consistent with training
        args.img = a["img"]
    enc = load_frozen_encoder(args.ckpt, device, **enc_kw)
    print(f"[dump] {os.path.basename(args.ckpt)} | encoder {enc_kw} | dataset={args.dataset}")

    x = get_batch(args, device)
    with torch.no_grad():
        tokens = enc(x)                                   # (B, n, d)
    z = tokens.reshape(-1, tokens.size(-1)).float()
    z = z - z.mean(dim=0, keepdim=True)
    C = (z.t() @ z) / z.size(0)                            # (d, d) covariance
    tr = torch.diagonal(C).sum()
    eff_rank = (tr * tr / (C.pow(2).sum() + 1e-12)).item()

    eig = torch.linalg.eigvalsh(C).flip(0).clamp(min=0)    # descending, non-negative
    top = eig[:args.topn]
    top_norm = (top / (top[0] + 1e-12))                    # normalized to the leading eigenvalue

    print(f"\neff_rank = {eff_rank:.2f}   (D = {enc_kw['d']}, tokens = {z.size(0)})")
    print("scree (top-%d eigenvalues, normalized) — paste this line:" % args.topn)
    print(json.dumps([round(v, 4) for v in top_norm.tolist()]))


if __name__ == "__main__":
    main()
