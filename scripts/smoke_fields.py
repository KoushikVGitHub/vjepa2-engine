"""Smoke test for src/data/fields.py on a real CAMELS field file.

Run on the pod after the data lands:
    python scripts/smoke_fields.py --field /workspace/data/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy
    # signed field -> transform asinh:
    python scripts/smoke_fields.py --field .../Maps_Vgas_...npy --transform asinh
    # with labels (needs the params file):
    python scripts/smoke_fields.py --field .../Maps_Mgas_...npy \
        --params /workspace/data/params_LH_IllustrisTNG.txt --min-std 0.006

Checks, in order:
  1. analyze_fields  -> raw-min (log10 vs asinh?) + std percentiles (pick min_std from ~p10)
  2. build a FieldMapDataset with the chosen min_std
  3. pull one batch through a DataLoader and ASSERT:
       - shape (B, 1, 256, 256), dtype float32, no NaN/Inf
       - post-standardization batch mean ~0, std ~1
       - if --params: label shape (B, 6) and sim-index alignment sanity
"""
import argparse
import os
import sys

import numpy as np
import torch

# make src/data importable no matter the CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
import fields  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True, help="path to one Maps_<field>_<suite>_LH_z=0.00.npy")
    ap.add_argument("--name", default="field")
    ap.add_argument("--transform", default="log10", choices=["log10", "asinh", "none"])
    ap.add_argument("--params", default=None, help="params_LH_<suite>.txt (optional, enables labels)")
    ap.add_argument("--min-std", type=float, default=None, help="curation threshold; omit to keep all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print("=" * 70, "\n[1] analyze_fields — pick transform + min_std\n", "=" * 70)
    fields.analyze_fields(args.field, transform=args.transform)

    print("\n" + "=" * 70, "\n[2] build FieldMapDataset (this runs the single offline pass)\n", "=" * 70)
    ds = fields.FieldMapDataset(
        args.field, name=args.name, transform=args.transform,
        min_std=args.min_std, params_path=args.params,
        return_params=args.params is not None,
    )
    print(f"len(ds)={len(ds)}  mean={ds.mean:.4f}  std={ds.std:.4f}  maps_per_sim={ds.maps_per_sim}")

    print("\n" + "=" * 70, "\n[3] one batch through a DataLoader — assertions\n", "=" * 70)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=True, pin_memory=True, drop_last=True,
    )
    batch = next(iter(loader))
    x, y = (batch if args.params else (batch, None))

    assert x.shape[1:] == (1, 256, 256), f"bad map shape {tuple(x.shape)}"
    assert x.dtype == torch.float32, f"bad dtype {x.dtype}"
    assert torch.isfinite(x).all(), "NaN/Inf in a curated+standardized map (bad transform?)"
    m, s = x.mean().item(), x.std().item()
    print(f"batch x: {tuple(x.shape)}  mean={m:+.3f} (~0)  std={s:.3f} (~1)  finite=OK")
    assert abs(m) < 0.5 and 0.5 < s < 1.7, "standardization looks off (per-batch is noisy, but not THIS far)"

    if y is not None:
        assert y.shape == (args.batch_size, 6), f"bad label shape {tuple(y.shape)}"
        print(f"batch y: {tuple(y.shape)}  Omega_m range [{y[:,0].min():.3f},{y[:,0].max():.3f}]"
              f"  sigma_8 range [{y[:,1].min():.3f},{y[:,1].max():.3f}]")
        # sanity: CAMELS LH priors are Omega_m in [0.1,0.5], sigma_8 in [0.6,1.0]
        assert 0.05 < y[:, 0].min() and y[:, 0].max() < 0.55, "Omega_m out of CAMELS prior — label misalignment?"

    print("\nSMOKE PASSED ✅")


if __name__ == "__main__":
    main()
