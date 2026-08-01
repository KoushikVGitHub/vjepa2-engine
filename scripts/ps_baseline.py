"""Track-1 diagnostic: locate the SSL probe against the classical INFORMATION CEILINGS.

Per CAMELS field map we compute two classical feature sets and ridge-regress each to
(Omega_m, sigma_8), on a SIM-LEVEL split (same discipline as the probe), then read the gaps:

  pk       : radially-averaged 2D power spectrum  -> the 2-POINT / GAUSSIAN information line
  moments  : mean, std, skew, (excess) kurtosis   -> a cheap NON-GAUSSIAN summary
  pk+mom   : both, concatenated                    -> the classical combined ceiling

How to read it against our SSL probe (Mgas: R^2 Omega_m ~0.50, sigma_8 ~0.31):

  SSL ~= pk           -> the encoder captured only 2-point info; we're stuck at the power-
                         spectrum plateau (understanding C5). Decorrelation can't fix this;
                         only a harder / non-Gaussian-sensitive pretext can (Track 3).
  SSL  > pk           -> the encoder already captures non-Gaussian structure; the collapse
                         fix was the real win, and there's genuine SSL value over 2-point.
  moments  > SSL      -> accessible non-Gaussian info the pretext is LEAVING on the table
                         (a trivial 4-number summary beats the learned features) -> strongest
                         motivation for Track 3, and a specific target to beat.

Computed on the SAME transform the encoder sees (log10), so it answers "what did the encoder
extract from ITS input", not a separate physical-units power spectrum. Feature/scale offsets
are affine and absorbed by ridge's own per-feature standardization, so per-field standardization
is omitted (it only shifts the DC bin / an overall constant).

Pure numpy -- no sklearn/scipy -- so it runs on the pod with zero extra installs.

  # in-suite only:
  python scripts/ps_baseline.py \
      --npy   /workspace/data/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy \
      --params /workspace/data/params_LH_IllustrisTNG.txt \
      --field Mgas

  # + CROSS-SUITE: fit the ridge on the --npy suite (train sims), then also report on a
  # held-out suite (ALL its maps) -- the classical foil for the SSL probe's ITNG->SIMBA drop:
  python scripts/ps_baseline.py \
      --npy    /workspace/data/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy \
      --params /workspace/data/params_LH_IllustrisTNG.txt \
      --heldout-npy    /workspace/data/Maps_Mgas_SIMBA_LH_z=0.00.npy \
      --heldout-params /workspace/data/params_LH_SIMBA.txt \
      --field Mgas
"""
import argparse
import os
import sys

import numpy as np

# Reuse the PROBE's exact split so pk is scored on identical sims (apples-to-apples). Importing
# probe pulls torch, but it makes the split a single source of truth -- no risk of a re-implemented
# split drifting from the probe's. scripts/test_ps_baseline_split.py proves the alignment on CPU.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from probe import sim_split


# ----------------------------------------------------------------- power spectrum
def build_radial_index(H, W, nbins):
    """Radial-bin index for an (H,W) FFT power map. Geometry is fixed across all maps, so this
    is computed ONCE and reused -- the per-map cost is then just an FFT + a bincount."""
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
    edges = np.linspace(0.0, float(np.hypot(cy, cx)), nbins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    counts = np.maximum(np.bincount(idx, minlength=nbins), 1)
    return idx, counts


def power_spectrum(img, idx, counts, nbins):
    "Radially-averaged 2D power spectrum P(k) of a real image."
    P = np.fft.fftshift(np.abs(np.fft.fft2(img)) ** 2).ravel()
    return np.bincount(idx, weights=P, minlength=nbins) / counts


def moments(img):
    "Pixel-value moments: [mean, std, skew, excess kurtosis]. skew/kurt carry non-Gaussianity."
    x = img.ravel()
    mu = x.mean()
    sd = x.std() + 1e-8
    z = (x - mu) / sd
    return np.array([mu, sd, (z ** 3).mean(), (z ** 4).mean() - 3.0])


def featurize(npy, params_path, transform, nbins, maps_per_sim, idx, counts, limit, tag):
    """Featurize one suite -> (PK, MOM, Y, N). Shared by the in-suite and held-out suites so
    both go through the identical FFT/moment/log10 pipeline (idx/counts assume matched H,W)."""
    maps = np.load(npy, mmap_mode="r")
    params = np.loadtxt(params_path, dtype=np.float64)
    N = len(maps) if limit <= 0 else min(len(maps), limit)
    H, W = maps.shape[1], maps.shape[2]
    assert idx.size == H * W, f"[{tag}] geometry {H}x{W} != prebuilt radial index (size {idx.size})"
    print(f"[{tag}] {N} maps @ {H}x{W}, transform={transform}")
    PK = np.zeros((N, nbins)); MOM = np.zeros((N, 4)); Y = np.zeros((N, 2))
    for i in range(N):
        m = maps[i].astype(np.float64)
        if transform == "log10":
            m = np.log10(np.clip(m, 1e-6, None))
        elif transform == "asinh":
            m = np.arcsinh(m / 1e-6)
        PK[i] = np.log10(power_spectrum(m, idx, counts, nbins) + 1e-12)
        MOM[i] = moments(m)
        Y[i] = params[i // maps_per_sim, :2]
        if i % 2000 == 0:
            print(f"  [{tag}] featurized {i}/{N}")
    return PK, MOM, Y, N


# ----------------------------------------------------------------- ridge (closed form, numpy)
def ridge_test(Xtr, ytr, Xva, yva, Xte, yte, alphas, Xho=None, yho=None):
    """Fit ridge on TRAIN, select alpha by best VAL R^2, then report R^2/RMSE on the held-out TEST.

    Returns (r2_test, rmse_test, alpha, r2_val, r2_ho). The reported number is the TEST R^2 -- the
    model-selection (val) set is NOT reported, so this is an honest generalization number directly
    comparable to the SSL probe's test R^2. Features standardized on TRAIN stats only; bias column
    UNPENALIZED; y is 1-D so alpha is selected per target. The fit (A,b) uses train only, so
    selecting alpha on val never leaks into the coefficients.

    If (Xho, yho) is given (a HELD-OUT SUITE), it is standardized by the SAME train stats and scored
    with the SAME fitted w -> the classical cross-suite number, the foil for the SSL probe's
    ITNG-trained-head-applied-to-SIMBA transfer. r2_ho is None when no held-out set is passed.
    """
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    aug = lambda X: np.hstack([(X - m) / s, np.ones((len(X), 1))])
    Xtr_, Xva_, Xte_ = aug(Xtr), aug(Xva), aug(Xte)
    D = Xtr_.shape[1]
    pen = np.eye(D); pen[-1, -1] = 0.0                     # don't penalize the bias term
    A, b = Xtr_.T @ Xtr_, Xtr_.T @ ytr
    ss_va = ((yva - yva.mean()) ** 2).sum()
    best_al, best_val = None, -np.inf
    for al in alphas:                                      # select alpha on VAL
        w = np.linalg.solve(A + al * pen, b)
        r2 = 1.0 - ((yva - Xva_ @ w) ** 2).sum() / ss_va
        if r2 > best_val:
            best_val, best_al = r2, float(al)
    w = np.linalg.solve(A + best_al * pen, b)              # report on held-out TEST
    resid = yte - Xte_ @ w
    ss_te = ((yte - yte.mean()) ** 2).sum()
    r2_te = 1.0 - (resid ** 2).sum() / ss_te
    r2_ho = None
    if Xho is not None:
        # held-out suite: standardize by TRAIN(in-suite) stats, score with the same w.
        rho = yho - aug(Xho) @ w
        r2_ho = 1.0 - (rho ** 2).sum() / ((yho - yho.mean()) ** 2).sum()
    return r2_te, float(np.sqrt((resid ** 2).mean())), best_al, best_val, r2_ho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, help="Maps_<field>_<suite>_LH_z=0.00.npy (IN-SUITE / fit)")
    ap.add_argument("--params", required=True, help="params txt, (n_sims, >=2): col0=Omega_m, col1=sigma_8")
    ap.add_argument("--heldout-npy", default="", help="optional held-out SUITE maps -> cross-suite drop")
    ap.add_argument("--heldout-params", default="", help="params txt for the held-out suite")
    ap.add_argument("--field", default="field")
    ap.add_argument("--nbins", type=int, default=32, help="P(k) radial bins")
    ap.add_argument("--maps-per-sim", type=int, default=15)
    ap.add_argument("--transform", default="log10", choices=["log10", "asinh", "none"])
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="DEPRECATED / ignored -- the split is now probe.sim_split (fixed 80/10/10).")
    ap.add_argument("--limit", type=int, default=0, help="cap #maps for a quick pass (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    probe_maps = np.load(args.npy, mmap_mode="r")
    H, W = probe_maps.shape[1], probe_maps.shape[2]
    idx, counts = build_radial_index(H, W, args.nbins)
    del probe_maps

    PK, MOM, Y, N = featurize(args.npy, args.params, args.transform, args.nbins,
                              args.maps_per_sim, idx, counts, args.limit, f"{args.field}/in-suite")

    heldout = bool(args.heldout_npy)
    if heldout:
        PKh, MOMh, Yh, Nh = featurize(args.heldout_npy, args.heldout_params, args.transform,
                                      args.nbins, args.maps_per_sim, idx, counts, args.limit,
                                      f"{args.field}/held-out")

    # sim-level split IDENTICAL to the probe (probe.sim_split, 80/10/10, seed=0): pk is fit on the
    # same train sims, alpha selected on the same val sims, and reported on EXACTLY the probe's
    # held-out TEST sims -> the pk number is now apples-to-apples with the SSL probe's test R^2.
    tr_idx, va_idx, te_idx = sim_split(N, maps_per_sim=args.maps_per_sim, seed=args.seed)
    print(f"[{args.field}] sim_split(seed={args.seed}): {len(tr_idx)} train / {len(va_idx)} val / "
          f"{len(te_idx)} test maps; reporting in-suite TEST R^2"
          + (f" + cross-suite on ALL {Nh} held-out maps\n" if heldout else "\n"))

    alphas = np.logspace(-3, 4, 15)
    feats = {"pk": (PK, PKh if heldout else None),
             "moments": (MOM, MOMh if heldout else None),
             "pk+moments": (np.hstack([PK, MOM]),
                            np.hstack([PKh, MOMh]) if heldout else None)}
    names = ["Omega_m", "sigma_8"]

    hdr = f"{'feature':<12}{'dim':>5}   {'R2(Om)':>8}{'R2(s8)':>8}"
    if heldout:
        hdr += f"   {'xR2(Om)':>8}{'xR2(s8)':>8}   {'dropOm':>7}{'drops8':>7}"
    hdr += f"   {'RMSE(Om)':>10}{'RMSE(s8)':>10}   (in-suite TEST" + (" | cross=SIMBA all)" if heldout else ")")
    print(hdr)
    print("-" * len(hdr))
    for fname, (X, Xh) in feats.items():
        r2s, xr2s, rmses = [], [], []
        for t in range(2):
            Xho = Xh if heldout else None
            yho = Yh[:, t] if heldout else None
            r2, rmse, _, _, r2_ho = ridge_test(X[tr_idx], Y[tr_idx, t], X[va_idx], Y[va_idx, t],
                                               X[te_idx], Y[te_idx, t], alphas, Xho, yho)
            r2s.append(r2); rmses.append(rmse)
            if heldout:
                xr2s.append(r2_ho)
        row = f"{fname:<12}{X.shape[1]:>5}   {r2s[0]:>8.3f}{r2s[1]:>8.3f}"
        if heldout:
            row += (f"   {xr2s[0]:>8.3f}{xr2s[1]:>8.3f}   "
                    f"{r2s[0]-xr2s[0]:>7.3f}{r2s[1]-xr2s[1]:>7.3f}")
        row += "   " + "".join(f"{r:>10.4f}" for r in rmses)
        print(row)

    print("-" * len(hdr))
    print("* in-suite TEST R^2 = comparable to run_probe.py's IN-SUITE R^2 (same sim_split seed 0).")
    if heldout:
        print("* xR2 = cross-suite R^2 (fit on in-suite train, applied to ALL held-out maps) = the")
        print("  classical foil for the SSL probe's HELD-OUT (SIMBA) R^2. The headline compares the")
        print("  SSL cross-suite DROP against pk's 'drop' column: SSL retains more -> transfer win.")


if __name__ == "__main__":
    main()
