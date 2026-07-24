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

  python scripts/ps_baseline.py \
      --npy   /workspace/data/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy \
      --params /workspace/data/params_LH_IllustrisTNG.txt \
      --field Mgas
"""
import argparse
import numpy as np


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


# ----------------------------------------------------------------- ridge (closed form, numpy)
def ridge_r2(Xtr, ytr, Xva, yva, alphas):
    """Fit ridge on train, pick alpha by best VAL R^2, return (r2, rmse, alpha).

    Features standardized on train stats; bias column left UNPENALIZED. y is 1-D (one target),
    so alpha is selected per target -- the fair per-parameter reading.
    """
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = np.hstack([(Xtr - m) / s, np.ones((len(Xtr), 1))])
    Xva = np.hstack([(Xva - m) / s, np.ones((len(Xva), 1))])
    D = Xtr.shape[1]
    pen = np.eye(D); pen[-1, -1] = 0.0                      # don't penalize the bias term
    A, b = Xtr.T @ Xtr, Xtr.T @ ytr
    ss_tot = ((yva - yva.mean()) ** 2).sum()
    best = (-np.inf, np.inf, None)
    for al in alphas:
        w = np.linalg.solve(A + al * pen, b)
        resid = yva - Xva @ w
        r2 = 1.0 - (resid ** 2).sum() / ss_tot
        if r2 > best[0]:
            best = (r2, float(np.sqrt((resid ** 2).mean())), float(al))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, help="Maps_<field>_<suite>_LH_z=0.00.npy")
    ap.add_argument("--params", required=True, help="params txt, (n_sims, >=2): col0=Omega_m, col1=sigma_8")
    ap.add_argument("--field", default="field")
    ap.add_argument("--nbins", type=int, default=32, help="P(k) radial bins")
    ap.add_argument("--maps-per-sim", type=int, default=15)
    ap.add_argument("--transform", default="log10", choices=["log10", "asinh", "none"])
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of SIMS held out")
    ap.add_argument("--limit", type=int, default=0, help="cap #maps for a quick pass (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    maps = np.load(args.npy, mmap_mode="r")
    params = np.loadtxt(args.params, dtype=np.float64)
    N = len(maps) if args.limit <= 0 else min(len(maps), args.limit)
    H, W = maps.shape[1], maps.shape[2]
    idx, counts = build_radial_index(H, W, args.nbins)
    print(f"[{args.field}] {N} maps @ {H}x{W}, {args.nbins} P(k) bins, transform={args.transform}")

    PK = np.zeros((N, args.nbins))
    MOM = np.zeros((N, 4))
    Y = np.zeros((N, 2))
    SIM = np.zeros(N, dtype=np.int64)
    for i in range(N):
        m = maps[i].astype(np.float64)
        if args.transform == "log10":
            m = np.log10(np.clip(m, 1e-6, None))
        elif args.transform == "asinh":
            m = np.arcsinh(m / 1e-6)
        PK[i] = np.log10(power_spectrum(m, idx, counts, args.nbins) + 1e-12)
        MOM[i] = moments(m)
        sim = i // args.maps_per_sim
        SIM[i], Y[i] = sim, params[sim, :2]
        if i % 2000 == 0:
            print(f"  featurized {i}/{N}")

    # sim-level split: whole simulations go entirely to train or val (no map from a val sim
    # ever leaks into train), matching the probe's split discipline.
    sims = np.unique(SIM)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sims)
    n_val = max(1, int(len(sims) * args.val_frac))
    val_sims = set(sims[:n_val].tolist())
    va = np.array([s in val_sims for s in SIM])
    tr = ~va
    print(f"[{args.field}] split: {tr.sum()} train / {va.sum()} val maps "
          f"({len(sims) - n_val}/{n_val} sims)\n")

    alphas = np.logspace(-3, 4, 15)
    feats = {"pk": PK, "moments": MOM, "pk+moments": np.hstack([PK, MOM])}
    names = ["Omega_m", "sigma_8"]

    print(f"{'feature':<12}{'dim':>5}   "
          f"{'R2(Om)':>8}{'R2(s8)':>8}   {'RMSE(Om)':>10}{'RMSE(s8)':>10}")
    print("-" * 66)
    for fname, X in feats.items():
        row = [f"{fname:<12}{X.shape[1]:>5}   "]
        rmses = []
        for t in range(2):
            r2, rmse, _ = ridge_r2(X[tr], Y[tr, t], X[va], Y[va, t], alphas)
            row.append(f"{r2:>8.3f}")
            rmses.append(rmse)
        row.append("   " + "".join(f"{r:>10.4f}" for r in rmses))
        print("".join(row))

    print("-" * 66)
    print(f"{'SSL probe*':<12}{'--':>5}   {0.500:>8.3f}{0.310:>8.3f}   "
          f"{'(Mgas ref)':>10}")
    print("\n* SSL reference = trained attentive probe on our VISReg+cov encoder (Mgas).")
    print("  Read:  SSL~=pk -> Gaussian plateau (Track 3 warranted);  SSL>pk -> real SSL gain;"
          "\n         moments>SSL -> non-Gaussian info the pretext is leaving on the table.")


if __name__ == "__main__":
    main()
