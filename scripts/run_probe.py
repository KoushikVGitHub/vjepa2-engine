"""Driver: train + evaluate the cosmology probe on the frozen ViT-L JEPA encoder.

Runs single-GPU. Rebuilds the encoder at the KEEPER's ViT-L config (d1024 / L24 / heads16,
img256 / patch16) so the saved context_encoder state_dict loads cleanly; load_frozen_encoder
RAISES if any encoder param is missing, which guards against a silently-random encoder from a
config mismatch. Split is at SIMULATION granularity (probe.sim_split) -- never by map.

Usage:
  python scripts/run_probe.py --ckpt /workspace/ckpt.pt --field Mgas --epochs 20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import Subset, DataLoader

from data.fields import FieldMapDataset
from probe import (load_frozen_encoder, ProbeHead, sim_split,
                   train_probe, eval_probe, latent_atlas)

# ViT-L encoder config -- MUST match the keeper run or the state_dict won't load.
ENC = dict(img=256, patch=16, d=1024, heads=16, layers=24)


def make_dataset(data_root, field, suite):
    npy = os.path.join(data_root, f"Maps_{field}_{suite}_LH_z=0.00.npy")
    params = os.path.join(data_root, f"params_LH_{suite}.txt")
    # log10 + min_std=0.05 must match the corpus config so the cached manifest is reused
    # (and, for keep-all fields, manifest is identity -> sim_split map indices == dataset positions).
    return FieldMapDataset(npy, name=field, transform="log10", min_std=0.05,
                           params_path=params, return_params=True, use_cache=True)


def loader(ds, idx, bs, shuffle, workers):
    return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=shuffle,
                      num_workers=workers, drop_last=False, pin_memory=True)


def _print_metrics(res):
    for k, v in res.items():
        print(f"  {k:9s}: Omega_m={v[0]:.4f}   sigma8={v[1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt.pt")
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--field", default="Mgas")
    ap.add_argument("--suite", default="IllustrisTNG", help="in-suite (pretraining) suite")
    ap.add_argument("--heldout", default="SIMBA", help="cross-suite robustness suite (if on disk)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-atlas", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = load_frozen_encoder(args.ckpt, device, **ENC)
    n_enc = sum(p.numel() for p in enc.parameters())
    print(f"[probe] frozen encoder loaded: {n_enc / 1e6:.1f}M params (ViT-L, {ENC})")

    ds = make_dataset(args.data_root, args.field, args.suite)
    tr_idx, va_idx, te_idx = sim_split(len(ds))
    print(f"[probe] {args.field}/{args.suite}: {len(ds)} maps -> "
          f"{len(tr_idx)} train / {len(va_idx)} val / {len(te_idx)} test (sim-level split)")

    tr = loader(ds, tr_idx, args.batch, True, args.workers)
    va = loader(ds, va_idx, 128, False, args.workers)
    te = loader(ds, te_idx, 128, False, args.workers)

    head = ProbeHead(d=ENC["d"]).to(device)
    train_probe(enc, head, tr, va, device, epochs=args.epochs)

    print(f"\n=== IN-SUITE ({args.suite}) ===")
    _print_metrics(eval_probe(enc, head, te, device))

    # Cross-suite robustness (challenge #2) -- only if the held-out suite is on disk.
    held_npy = os.path.join(args.data_root, f"Maps_{args.field}_{args.heldout}_LH_z=0.00.npy")
    if os.path.exists(held_npy):
        hds = make_dataset(args.data_root, args.field, args.heldout)
        hte = loader(hds, list(range(len(hds))), 128, False, args.workers)
        print(f"\n=== HELD-OUT ({args.heldout}) = cross-suite robustness ===")
        _print_metrics(eval_probe(enc, head, hte, device))
    else:
        print(f"\n[probe] held-out suite {args.heldout} not on disk -- skipping cross-suite eval.")

    if not args.no_atlas:
        try:
            latent_atlas(enc, te, device, out_dir="notes/atlas")
            print("[probe] latent atlas -> notes/atlas/")
        except Exception as e:
            print(f"[probe] atlas skipped ({type(e).__name__}: {e}); "
                  f"pip install matplotlib scikit-learn to enable.")


if __name__ == "__main__":
    main()
