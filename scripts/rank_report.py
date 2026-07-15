"""Diagnostic: is the frozen encoder's PROBE-FACING representation rank-limited?

Training logs effective rank over `full_flat` -- every patch token of every image, flattened
(jepa_loss._forward_lejepa). The probe never sees that cloud: it POOLS each image's tokens into
one vector per map. CAMELS fields are spatially smooth, so tokens within an image are highly
redundant -- the token cloud can carry rank ~38 while the pooled per-image vectors that actually
feed the probe carry far less. If pooled rank is the binding constraint, more pretraining steps
buy little and the fix is to regularize the pooled representation instead.

Reports, on a frozen checkpoint (no training):
  1. token eff_rank      -- reproduces the training-time number (sanity check vs the run log)
  2. pooled eff_rank     -- the number that was never measured; the probe's real input geometry
  3. pooled PCA spectrum -- dims needed for 90/95/99% variance
  4. linear probe R^2(k) -- closed-form ridge on the top-k PCs, sim-level split.
                            Says how many dims carry COSMOLOGY vs nuisance, and gives a fast
                            R^2 estimate to compare against the trained attentive probe.

Usage:
  python scripts/rank_report.py --ckpt /workspace/ckpt.pt --field Mgas --n 3000
  python scripts/rank_report.py --ckpt /workspace/ckpt_10k.pt --field Mgas --n 3000   # after 10k
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from data.fields import FieldMapDataset
from probe import load_frozen_encoder, sim_split

# MUST match the keeper run or the state_dict won't load (load_frozen_encoder raises).
ENC = dict(img=256, patch=16, d=1024, heads=16, layers=24)

# CAMELS LH prior midpoints -- probe.py normalizes targets by these; mirrored here so the
# reported R^2 is on the same footing. R^2 is scale-invariant, so this only affects ridge alpha.
TARGET_MEAN = torch.tensor([0.3, 0.8])
TARGET_STD = torch.tensor([0.1, 0.1])


def effective_rank(C: torch.Tensor) -> float:
    """Participation ratio tr(C)^2 / ||C||_F^2 in [1, d]. Same formula the training loop logs."""
    tr = torch.diagonal(C).sum()
    return (tr * tr / (C.pow(2).sum() + 1e-12)).item()


@torch.no_grad()
def extract(encoder, loader, device, d):
    """One pass: stream token + within-image covariance (d x d each), keep pooled vectors."""
    # Token stats stream in float64 -- B*n tokens never fits in memory, but the (d,d) outer-product
    # accumulator does. C = E[zz^T] - mu mu^T at the end.
    sum_z = torch.zeros(d, dtype=torch.float64, device=device)
    sum_zz = torch.zeros(d, d, dtype=torch.float64, device=device)
    n_tokens = 0

    # Within-image covariance: deviations of each token from ITS OWN image's mean, pooled over
    # images. By the law of total covariance, C_token = C_between + C_within exactly, where
    # C_between = cov of the pooled per-image vectors. Isolating C_within explains whether an
    # equal token/pooled participation ratio means "no within-image variation" (spatial collapse)
    # or "within-image variation shares the between-image subspace" (a real, benign structure).
    sum_ww = torch.zeros(d, d, dtype=torch.float64, device=device)

    pooled, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            tokens = encoder(x)                       # (B, n, d)
        tokens = tokens.float()

        z = tokens.reshape(-1, d).double()            # (B*n, d) -- the training-time view
        sum_z += z.sum(dim=0)
        sum_zz += z.t() @ z
        n_tokens += z.size(0)

        img_mean = tokens.mean(dim=1, keepdim=True)             # (B, 1, d) -- the probe's view
        w = (tokens - img_mean).reshape(-1, d).double()         # within-image deviations
        sum_ww += w.t() @ w

        pooled.append(img_mean.squeeze(1).cpu())      # (B, d) -- the probe-facing view
        labels.append(y[:, :2].clone())               # cols 0,1 = Omega_m, sigma_8

    mu = sum_z / n_tokens
    token_cov = sum_zz / n_tokens - torch.outer(mu, mu)
    within_cov = sum_ww / n_tokens
    # Back to CPU: pooled/labels are already CPU, and the downstream eigh/svd/ridge all run there.
    # (Mixing a CUDA within_cov with a CPU pooled_cov in subspace_alignment is a device-mismatch
    # crash that a CPU-only dev box never reproduces.)
    return token_cov.cpu(), within_cov.cpu(), torch.cat(pooled), torch.cat(labels)


def subspace_alignment(A: torch.Tensor, B: torch.Tensor, k: int = 10) -> float:
    """Mean squared cosine of the principal angles between the top-k eigenspaces of A and B.

    1.0 = the two covariances' dominant subspaces coincide; ~k/d = unrelated. Distinguishes
    "within-image variation is proportional to between-image variation" (aligned -> equal
    participation ratios) from a coincidence.
    """
    _, VA = torch.linalg.eigh(A)
    _, VB = torch.linalg.eigh(B)
    QA, QB = VA[:, -k:], VB[:, -k:]          # eigh returns ASCENDING eigenvalues -> take the tail
    return ((QA.t() @ QB) ** 2).sum().item() / k


def ridge_r2(pcs, y, tr_idx, te_idx, ks, alpha=1e-2):
    """Closed-form ridge on the top-k PCs; R^2 on a held-out SIM-level split, per target."""
    out = {}
    for k in ks:
        X_tr, X_te = pcs[tr_idx, :k], pcs[te_idx, :k]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Standardize features on TRAIN stats only, and add a bias column.
        m, s = X_tr.mean(0, keepdim=True), X_tr.std(0, keepdim=True).clamp_min(1e-8)
        X_tr, X_te = (X_tr - m) / s, (X_te - m) / s
        X_tr = torch.cat([X_tr, torch.ones(len(X_tr), 1, dtype=X_tr.dtype)], 1)
        X_te = torch.cat([X_te, torch.ones(len(X_te), 1, dtype=X_te.dtype)], 1)

        A = X_tr.t() @ X_tr + alpha * torch.eye(X_tr.size(1), dtype=X_tr.dtype)
        w = torch.linalg.solve(A, X_tr.t() @ y_tr)
        resid = ((y_te - X_te @ w) ** 2).sum(0)
        total = ((y_te - y_tr.mean(0, keepdim=True)) ** 2).sum(0)
        out[k] = (1 - resid / total).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt.pt")
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--field", default="Mgas")
    ap.add_argument("--suite", default="IllustrisTNG")
    ap.add_argument("--n", type=int, default=3000, help="maps to embed (multiple of 15 = whole sims)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_frozen_encoder(args.ckpt, device, **ENC)
    print(f"[rank] encoder loaded from {args.ckpt}")

    npy = os.path.join(args.data_root, f"Maps_{args.field}_{args.suite}_LH_z=0.00.npy")
    params = os.path.join(args.data_root, f"params_LH_{args.suite}.txt")
    # augment=False (default) -> deterministic views, same as probe train/eval.
    ds = FieldMapDataset(npy, name=args.field, transform="log10", min_std=0.05,
                         params_path=params, return_params=True, use_cache=True)

    # Take whole simulations from the front so the //15 sim mapping stays intact.
    n = min(args.n, len(ds)) // 15 * 15
    loader = DataLoader(Subset(ds, range(n)), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    token_cov, within_cov, pooled, y = extract(encoder, loader, device, ENC["d"])
    print(f"[rank] embedded {n} maps -> pooled {tuple(pooled.shape)}")

    # ---- 1 & 2: the comparison this script exists for.
    tok_rank = effective_rank(token_cov)
    Z = (pooled - pooled.mean(0, keepdim=True)).double()
    pooled_cov = (Z.t() @ Z) / Z.size(0)
    pool_rank = effective_rank(pooled_cov)

    print("\n=== effective rank (participation ratio, d=1024) ===")
    print(f"  tokens (training logs this) : {tok_rank:7.2f}")
    print(f"  pooled (the probe sees this): {pool_rank:7.2f}")
    if pool_rank < 0.5 * tok_rank:
        print("  -> pooled rank is MUCH lower: pooling is the bottleneck, not pretraining length.")
        print("     More steps raise token rank the probe never consumes. Regularize the POOLED rep.")
    else:
        print("  -> pooled rank tracks token rank: pooling is not throwing the geometry away.")

    # ---- 2b: WHY they match. C_token = C_between + C_within (law of total covariance), and the
    # participation ratio is scale-invariant -- so equal ranks mean either C_within ~ 0 (every
    # patch of an image maps to the same vector = spatial collapse) or C_within is proportional
    # to C_between (same subspace, different scale = benign, and physically sensible if patches
    # vary within an image along the same feature axes that cosmology varies between images).
    r = (torch.diagonal(within_cov).sum() / torch.diagonal(pooled_cov).sum()).item()
    align = subspace_alignment(within_cov, pooled_cov, k=10)
    print("\n=== within-image vs between-image variance ===")
    print(f"  within/between total variance ratio : {r:8.3f}")
    print(f"  within eff_rank                     : {effective_rank(within_cov):8.2f}")
    print(f"  top-10 subspace alignment [0..1]    : {align:8.3f}")
    if r < 0.05:
        print("  -> within-image variance is ~0: every patch embeds to its image's vector.")
        print("     SPATIAL COLLAPSE -- the encoder is ignoring position; masked prediction is trivial.")
    elif align > 0.5:
        print("  -> within- and between-image variation share a subspace (hence equal ranks).")
        print("     Benign: patches vary along the same feature axes that cosmology varies along.")
    else:
        print("  -> substantial within-image variance in a DIFFERENT subspace; equal ranks are then")
        print("     a coincidence of the participation ratio -- treat the pooled verdict with care.")

    # ---- 3: how many pooled dims carry variance at all.
    _, S, V = torch.linalg.svd(Z, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    cum = torch.cumsum(evr, 0)
    print("\n=== pooled PCA spectrum ===")
    print("  top-10 explained variance:", " ".join(f"{v:.3f}" for v in evr[:10]))
    for frac in (0.90, 0.95, 0.99):
        print(f"  dims for {frac:.0%} variance: {int((cum < frac).sum()) + 1}")

    # ---- 4: how many of those dims are COSMOLOGY (vs nuisance variance).
    pcs = Z @ V.t()                                    # (N, d) principal-component scores
    y_n = ((y - TARGET_MEAN) / TARGET_STD).double()
    tr_idx, _, te_idx = sim_split(n, maps_per_sim=15)
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= min(len(tr_idx), pcs.size(1))]
    r2 = ridge_r2(pcs, y_n, tr_idx, te_idx, ks)

    print("\n=== linear ridge probe on top-k PCs (sim-level split, held-out R^2) ===")
    print("     k   Omega_m    sigma_8")
    for k in ks:
        print(f"  {k:4d}   {r2[k][0]:7.3f}    {r2[k][1]:7.3f}")
    print("\n  Reference: trained attentive probe = Omega_m 0.50 / sigma_8 0.31 (1000-step ckpt).")
    print("  If R^2 saturates by small k, cosmology lives in a few dims and extra rank is nuisance.")


if __name__ == "__main__":
    main()
