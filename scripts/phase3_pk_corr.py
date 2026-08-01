"""Phase-3 fairness check on the pk foil: is the SSL encoder's cross-suite edge INFORMATION or
just CALIBRATION?

ps_baseline.py reports cross-suite R^2 only. R^2 punishes a miscalibrated-but-informative
predictor exactly like an uninformative one, so "SSL +0.18 vs pk -0.52" could mean either
(a) SSL genuinely carries more suite-invariant cosmology, or (b) both carry the same signal and
SSL merely happens to land on a better scale. Pearson r settles it: the best affine rescaling of
a single predictor achieves R^2 == r^2, so r^2 is the CEILING any recalibration of pk could reach.

Same feature code, same ridge, same sim-level split as ps_baseline / phase3_diag, so the numbers
drop straight into the Phase-3 table next to the SSL rows.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from probe import sim_split                                          # noqa: E402
from ps_baseline import build_radial_index, featurize                # noqa: E402

ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
PARAMS = ["Omega_m", "sigma8"]


def fit_ridge(Xtr, ytr, Xva, yva):
    """Ridge with unpenalised bias, alpha chosen on val R^2 -- ps_baseline's protocol."""
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    aug = lambda X, mm=m, ss=s: np.hstack([(X - mm) / ss, np.ones((len(X), 1))])
    Xtr_, Xva_ = aug(Xtr), aug(Xva)
    D = Xtr_.shape[1]
    pen = np.eye(D)
    pen[-1, -1] = 0.0
    A, b = Xtr_.T @ Xtr_, Xtr_.T @ ytr
    ss_va = ((yva - yva.mean(0)) ** 2).sum(0)
    best_w, best_sc, best_al = None, -np.inf, None
    for al in ALPHAS:
        w = np.linalg.solve(A + al * pen, b)
        r2 = 1.0 - ((yva - Xva_ @ w) ** 2).sum(0) / ss_va
        if r2.mean() > best_sc:
            best_w, best_sc, best_al = w, r2.mean(), al
    return best_w, m, s, best_al


def metrics(y, p):
    out = {}
    for j, name in enumerate(PARAMS):
        yt, pt = y[:, j], p[:, j]
        r2 = 1.0 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        r = np.corrcoef(yt, pt)[0, 1] if pt.std() > 1e-12 else 0.0
        out[name] = dict(r2=r2, r=r, r2_affine=r ** 2,
                         bias=(pt.mean() - yt.mean()) / yt.std(), spread=pt.std() / yt.std())
    return out


def fmt(m, label):
    s = f"  {label:<34s}"
    for name in PARAMS:
        d = m[name]
        s += (f"| {name}: R2={d['r2']:+7.3f} r={d['r']:+.3f} "
              f"r2aff={d['r2_affine']:.3f} bias={d['bias']:+.2f}s spread={d['spread']:.2f} ")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--heldout-npy", required=True)
    ap.add_argument("--heldout-params", required=True)
    ap.add_argument("--field", default="Mgas")
    ap.add_argument("--nbins", type=int, default=32)
    ap.add_argument("--maps-per-sim", type=int, default=15)
    ap.add_argument("--transform", default="log10")
    args = ap.parse_args()

    idx, counts = build_radial_index(256, 256, args.nbins)
    PK, MOM, Y, N = featurize(args.npy, args.params, args.transform, args.nbins,
                              args.maps_per_sim, idx, counts, 0, "in-suite")
    PKh, MOMh, Yh, Nh = featurize(args.heldout_npy, args.heldout_params, args.transform,
                                  args.nbins, args.maps_per_sim, idx, counts, 0, "held-out")
    tr, va, te = sim_split(N, args.maps_per_sim)

    for name, X, Xh in [("pk", PK, PKh), ("pk+moments", np.hstack([PK, MOM]), np.hstack([PKh, MOMh]))]:
        w, m, s, al = fit_ridge(X[tr], Y[tr], X[va], Y[va])
        aug = lambda Z, mm, ss: np.hstack([(Z - mm) / ss, np.ones((len(Z), 1))])
        print(f"\n[{name}] alpha={al:g}  dim={X.shape[1]}")
        print(fmt(metrics(Y[te], aug(X[te], m, s) @ w), "in-suite TEST"))
        print(fmt(metrics(Yh, aug(Xh, m, s) @ w), "x-suite (src-norm)"))
        mh, sh = Xh.mean(0), Xh.std(0) + 1e-8
        print(fmt(metrics(Yh, aug(Xh, mh, sh) @ w), "x-suite (self-norm)"))


if __name__ == "__main__":
    main()
