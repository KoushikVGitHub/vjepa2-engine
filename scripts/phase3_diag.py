"""Phase-3 diagnosis: discriminate readout-failure (H1/H3/H5) from representation-failure (H2/H4).

Runs entirely on the pooled features dumped by phase3_feats.py (CPU, seconds), so all the
controls share ONE frozen-encoder pass and one split.

Experiments (labels match the Phase-3 brief):
  B  ridge readout        fit ITNG -> apply SIMBA, ridge (regularised, fixed pool) vs the
                          MLP+attentive-pool head. Removes the head-overfit confound.
  A' SIMBA oracle (ridge) fit SIMBA -> test SIMBA. Upper bound: does the ITNG-trained encoder
                          even CONTAIN SIMBA cosmology? (companion to the GPU run of exp. A)
  R  reverse transfer     fit SIMBA -> apply ITNG.
  E  few-shot             fit on N SIMBA sims; and the cheaper variant, ITNG-fit weights with
                          only an affine recalibration estimated from N SIMBA sims.
  D  shift geometry       how out-of-distribution SIMBA features are, and whether the
                          cosmology DIRECTION itself moves between suites.

Two diagnostics do the real discriminating work and are reported for every arm:
  * Pearson r on the target suite. Best-possible affine recalibration of a single predictor
    gives R2 == r^2, so r^2 is the CEILING any rescaling fix could reach. r^2 high with R2
    negative => the information is there and the readout is merely miscalibrated (H1/H3/H5).
    r^2 ~ 0 => the information is not linearly there at all (H2/H4).
  * bias/sigma_y => how much of the negative R2 is a constant offset vs destroyed ranking.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from probe import sim_split                      # noqa: E402  (identical split to run_probe)

ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
PARAMS = ["Omega_m", "sigma8"]


def load(feat_dir, tag, stage, suite):
    z = np.load(os.path.join(feat_dir, f"{tag}_{stage}_{suite}.npz"))
    return z["X"].astype(np.float64), z["Y"][:, :2].astype(np.float64)


def standardizer(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def ridge_fit(X, y, alpha):
    """Closed-form ridge on already-standardised X; intercept handled by centering y."""
    n, d = X.shape
    ybar = y.mean(0)
    A = X.T @ X + alpha * np.eye(d)
    W = np.linalg.solve(A, X.T @ (y - ybar))
    return W, ybar


def predict(X, W, ybar):
    return X @ W + ybar


def metrics(y, p):
    out = {}
    for j, name in enumerate(PARAMS):
        yt, pt = y[:, j], p[:, j]
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        r2 = 1.0 - ((yt - pt) ** 2).sum() / ss_tot
        r = np.corrcoef(yt, pt)[0, 1] if pt.std() > 1e-12 else 0.0
        out[name] = dict(r2=r2, r=r, r2_affine=r ** 2,
                         bias=(pt.mean() - yt.mean()) / yt.std(),
                         spread=pt.std() / yt.std())
    return out


def fmt(m, label):
    s = f"  {label:<34s}"
    for name in PARAMS:
        d = m[name]
        s += (f"| {name}: R2={d['r2']:+7.3f} r={d['r']:+.3f} "
              f"r2aff={d['r2_affine']:.3f} bias={d['bias']:+.2f}s spread={d['spread']:.2f} ")
    return s


def pick_alpha(Xtr, ytr, Xva, yva):
    """One alpha for the pair, chosen by mean val R2 -- same protocol as a pk ridge sweep."""
    best, best_a = -1e18, ALPHAS[0]
    for a in ALPHAS:
        W, yb = ridge_fit(Xtr, ytr, a)
        m = metrics(yva, predict(Xva, W, yb))
        score = np.mean([m[p]["r2"] for p in PARAMS])
        if score > best:
            best, best_a = score, a
    return best_a, best


def arm(Xs, ys, Xt, yt_, tr, va, name, target_all=True):
    """Fit ridge on source suite train sims; report source-test + target-suite metrics."""
    mu, sd = standardizer(Xs[tr])
    Ztr, Zva = (Xs[tr] - mu) / sd, (Xs[va] - mu) / sd
    a, _ = pick_alpha(Ztr, ys[tr], Zva, ys[va])
    W, yb = ridge_fit(Ztr, ys[tr], a)
    print(f"\n[{name}] alpha={a:g}")
    return W, yb, mu, sd, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="/dev/shm/feat")
    ap.add_argument("--tag", default="conv_s0")
    ap.add_argument("--stage", default="encoder")
    ap.add_argument("--src", default="IllustrisTNG")
    ap.add_argument("--tgt", default="SIMBA")
    args = ap.parse_args()

    Xs, ys = load(args.feat_dir, args.tag, args.stage, args.src)
    Xt, yt = load(args.feat_dir, args.tag, args.stage, args.tgt)
    tr, va, te = sim_split(len(Xs))
    ttr, tva, tte = sim_split(len(Xt))
    print(f"=== Phase-3 diagnosis | tag={args.tag} stage={args.stage} "
          f"| {args.src} {Xs.shape} -> {args.tgt} {Xt.shape} ===")
    print(f"features = mean+std pooled tokens (fixed pool), ridge readout, "
          f"split {len(tr)}/{len(va)}/{len(te)} maps (sim-level)")

    # ---------------------------------------------------------------- B: ridge, src -> tgt
    W, yb, mu, sd, a = arm(Xs, ys, Xt, yt, tr, va, "B  ridge readout (fit %s)" % args.src)
    print(fmt(metrics(ys[te], predict((Xs[te] - mu) / sd, W, yb)),
              f"in-suite {args.src} TEST"))
    print(fmt(metrics(yt, predict((Xt - mu) / sd, W, yb)),
              f"x-suite {args.tgt} (src-norm)"))
    mu_t, sd_t = standardizer(Xt)
    print(fmt(metrics(yt, predict((Xt - mu_t) / sd_t, W, yb)),
              f"x-suite {args.tgt} (self-norm)"))

    # ---------------------------------------------------------------- A': SIMBA oracle (ridge)
    Wt, ybt, mut, sdt, at = arm(Xt, yt, Xt, yt, ttr, tva, "A' oracle (fit %s)" % args.tgt)
    print(fmt(metrics(yt[tte], predict((Xt[tte] - mut) / sdt, Wt, ybt)),
              f"in-suite {args.tgt} TEST"))
    print(fmt(metrics(ys, predict((Xs - mut) / sdt, Wt, ybt)),
              f"R  reverse -> {args.src} (src-norm)"))
    mu_s_all, sd_s_all = standardizer(Xs)
    print(fmt(metrics(ys, predict((Xs - mu_s_all) / sd_s_all, Wt, ybt)),
              f"R  reverse -> {args.src} (self-norm)"))

    # ---------------------------------------------------------------- E: few-shot on target
    print("\n[E  few-shot adaptation on %s]  (test = %s held-out sims)" % (args.tgt, args.tgt))
    rng = np.random.RandomState(0)
    tgt_train_sims = np.unique(np.array(ttr) // 15)
    for n in [1, 5, 20, 100, 800]:
        if n > len(tgt_train_sims):
            continue
        r2s_scratch, r2s_recal = [], []
        reps = 5 if n <= 100 else 1
        for rep in range(reps):
            sims = rng.choice(tgt_train_sims, size=n, replace=False)
            idx = np.array([s * 15 + k for s in sims for k in range(15)])
            # (a) ridge fit from scratch on N target sims
            mn, sdn = standardizer(Xt[idx]) if n >= 5 else (mut, sdt)
            Zn = (Xt[idx] - mn) / sdn
            best_r2, bw = None, None
            for al in ALPHAS:                      # tiny N -> pick alpha on the target TEST-free val
                Wn, ybn = ridge_fit(Zn, yt[idx], al)
                mv = metrics(yt[tva], predict((Xt[tva] - mn) / sdn, Wn, ybn))
                sc = np.mean([mv[p]["r2"] for p in PARAMS])
                if best_r2 is None or sc > best_r2:
                    best_r2, bw = sc, (Wn, ybn, mn, sdn)
            Wn, ybn, mn, sdn = bw
            m = metrics(yt[tte], predict((Xt[tte] - mn) / sdn, Wn, ybn))
            r2s_scratch.append([m[p]["r2"] for p in PARAMS])
            # (b) keep source weights, fit only a per-param affine recalibration on N target sims
            p_cal = predict((Xt[idx] - mu) / sd, W, yb)
            p_te = predict((Xt[tte] - mu) / sd, W, yb)
            rec = np.zeros_like(p_te)
            for j in range(2):
                A = np.polyfit(p_cal[:, j], yt[idx][:, j], 1)
                rec[:, j] = A[0] * p_te[:, j] + A[1]
            mr = metrics(yt[tte], rec)
            r2s_recal.append([mr[p]["r2"] for p in PARAMS])
        sc = np.mean(r2s_scratch, axis=0)
        rc = np.mean(r2s_recal, axis=0)
        print(f"  N={n:>4d} sims | scratch-ridge R2 Om={sc[0]:+.3f} s8={sc[1]:+.3f} "
              f"| src-weights+affine-recal R2 Om={rc[0]:+.3f} s8={rc[1]:+.3f}")

    # ---------------------------------------------------------------- D: shift geometry
    print("\n[D  representation shift %s vs %s]" % (args.src, args.tgt))
    shift = np.abs(mu_t - mu) / sd
    print(f"  per-dim mean shift |dmu|/sigma_src : median={np.median(shift):.2f} "
          f"p90={np.percentile(shift, 90):.2f} max={shift.max():.2f} "
          f"| frac dims >1sigma = {np.mean(shift > 1):.2f}")
    scale = sd_t / sd
    print(f"  per-dim scale ratio sigma_tgt/sigma_src: median={np.median(scale):.2f} "
          f"p10={np.percentile(scale, 10):.2f} p90={np.percentile(scale, 90):.2f}")

    # suite identity: how linearly separable are the two suites in feature space?
    Zs, Zt = (Xs - mu) / sd, (Xt - mu) / sd
    ns = min(len(Zs), len(Zt))
    Xc = np.vstack([Zs[:ns], Zt[:ns]])
    yc = np.concatenate([np.zeros(ns), np.ones(ns)])
    half = np.arange(ns)
    trc = np.concatenate([half[: ns // 2], ns + half[: ns // 2]])
    tec = np.concatenate([half[ns // 2:], ns + half[ns // 2:]])
    Wc, ybc = ridge_fit(Xc[trc], yc[trc][:, None], 100.0)
    acc = np.mean(((Xc[tec] @ Wc + ybc)[:, 0] > 0.5) == (yc[tec] > 0.5))
    print(f"  suite-identity linear probe accuracy      : {acc:.3f}  (0.5 = suites indistinguishable)")

    # top-k PCA subspace overlap (well-defined across different sample sets)
    k = 50
    Us = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)[2][:k].T
    Ut = np.linalg.svd(Zt - Zt.mean(0), full_matrices=False)[2][:k].T
    ov = np.linalg.norm(Us.T @ Ut, "fro") ** 2 / k
    print(f"  top-{k} PCA subspace overlap              : {ov:.3f}  (1 = identical subspace)")

    # does the COSMOLOGY DIRECTION itself move? (both fitted in the same src-standardised space)
    Wsrc, _ = ridge_fit(Zs[tr], ys[tr], a)
    Wtgt, _ = ridge_fit(Zt[ttr], yt[ttr], a)
    for j, name in enumerate(PARAMS):
        c = (Wsrc[:, j] @ Wtgt[:, j]) / (np.linalg.norm(Wsrc[:, j]) * np.linalg.norm(Wtgt[:, j]))
        print(f"  cos(w_{args.src}, w_{args.tgt}) for {name:<8s}: {c:+.3f}  "
              f"(1 = same readout direction, 0 = unrelated)")


if __name__ == "__main__":
    main()
