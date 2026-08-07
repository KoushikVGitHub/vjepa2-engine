#!/usr/bin/env python3
"""
dof_probe.py -- Measure the degrees of freedom of CAMELS 2D maps so that every
architectural choice (embedding dim, depth, patch size) can be grounded in
source truth instead of inherited from V-JEPA defaults.

Per suite, and per patch size, it reports three things:

  1. DATA INTRINSIC DIMENSION  -- how many DOF a map actually has.
       TwoNN (Facco 2017) + MLE (Levina-Bickel 2004) + PCA participation ratio.
       Grounds: encoder width (edim ~ a few x ID) and patch size (the scale
       where per-token structure is richest).

  2. TASK-RELEVANT DOF         -- how many feature directions carry Omega_m/sigma_8.
       PCA -> ridge k95 (components to reach 95% of the linear R^2 ceiling)
       + CCA canonical correlations.
       Grounds: the *useful* rank; if this is ~5 and edim is 1024, isotropy
       regularisation has ~1000 spare dims to hoard suite-specific nuisance in.

  3. SUITE-SHIFT DOF (>=2 suites) -- the subspace that moves ITNG<->SIMBA and how
       much it overlaps the cosmology-predictive subspace. Cross-suite transfer
       needs the retained subspace orthogonal to this one.

CPU-only, self-contained (numpy + sklearn). Reads the staged .npy maps and the
matching params_LH_*.txt. Designed to run on a cheap CPU pod mounting the
eur-is-1 volume -- same pattern as the SIMBA data pull.

Example (single suite):
  python dof_probe.py \
      --maps /workspace/data/Maps_Mgas_SIMBA_LH_z=0.00.npy:SIMBA \
      --params /workspace/data/params_LH_SIMBA.txt \
      --patch-sizes 8,16,32 --subsample 2000 --out dof_simba.json

Example (both suites -> also computes suite-shift DOF):
  python dof_probe.py \
      --maps /workspace/data/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy:ITNG \
             /workspace/data/Maps_Mgas_SIMBA_LH_z=0.00.npy:SIMBA \
      --params /workspace/data/params_LH_IllustrisTNG.txt \
               /workspace/data/params_LH_SIMBA.txt \
      --patch-sizes 8,16,32 --subsample 2000 --out dof_both.json

Assumptions (all overridable / verifiable):
  * CAMELS 2D LH maps are ordered block-wise: map i belongs to sim i//maps_per_sim,
    so labels are aligned with np.repeat(params, maps_per_sim). --maps-per-sim
    controls it (default 15). The label range sanity-check will warn if the
    alignment looks wrong.
  * Mgas maps span many orders of magnitude -> log10 by default (--no-log to skip).
  * params columns are [Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2]; only the
    first two are the readout target (--label-cols to change).
"""
import argparse
import json
import sys
import time

import numpy as np


# -----------------------------------------------------------------------------
# intrinsic-dimension estimators
# -----------------------------------------------------------------------------
def _knn_dists(X, k):
    """Distances to the k nearest neighbours (self excluded), shape (n, k)."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, 1:]  # drop the self-distance (0)


def twonn(X, discard_frac=0.1):
    """TwoNN intrinsic dimension (Facco et al. 2017).

    Uses the ratio mu = r2/r1 of the two nearest-neighbour distances. Under a
    locally-uniform density F(mu) = 1 - mu^{-d}, so a through-origin fit of
    -log(1-F) against log(mu) has slope d. The largest mu are discarded as they
    are the most sensitive to density inhomogeneity.
    """
    d = _knn_dists(X, 2)
    r1, r2 = d[:, 0], d[:, 1]
    good = r1 > 0
    mu = np.sort(r2[good] / r1[good])
    n = len(mu)
    cut = max(1, int(n * (1 - discard_frac)))
    mu = mu[:cut]
    # empirical CDF value assigned to each sorted mu
    F = np.arange(1, len(mu) + 1) / n
    x = np.log(mu)
    y = -np.log(np.clip(1 - F, 1e-12, None))
    return float(np.sum(x * y) / np.sum(x * x))


def mle_id(X, k=10):
    """Levina-Bickel maximum-likelihood intrinsic dimension, averaged over points.

    m_k(x) = [ (1/(k-1)) sum_{j=1..k-1} log(T_k(x)/T_j(x)) ]^{-1}, where T_j is the
    distance to the j-th neighbour. Returns the mean of m_k(x) over all x.
    """
    d = _knn_dists(X, k)                     # (n, k)
    Tk = d[:, -1][:, None]                    # k-th neighbour
    Tj = d[:, :-1]                            # neighbours 1..k-1
    ok = (Tj > 0).all(axis=1) & (Tk[:, 0] > 0)
    sum_log = np.log(Tk[ok] / Tj[ok]).sum(axis=1)
    sum_log = np.clip(sum_log, 1e-12, None)
    m = (k - 1) / sum_log
    return float(np.mean(m))


def pca_stats(X, max_svd_dim=8192):
    """PCA participation ratio + components needed for 90/95/99% variance."""
    Xc = X - X.mean(0)
    if Xc.shape[1] > max_svd_dim:
        # random projection keeps pairwise structure; keeps SVD tractable
        rng = np.random.default_rng(0)
        R = rng.standard_normal((Xc.shape[1], max_svd_dim)) / np.sqrt(max_svd_dim)
        Xc = Xc @ R
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s ** 2) / max(1, len(X) - 1)
    lam = lam[lam > 0]
    pr = float((lam.sum() ** 2) / np.square(lam).sum())
    cum = np.cumsum(lam) / lam.sum()

    def ncomp(t):
        return int(np.searchsorted(cum, t) + 1)

    return dict(participation_ratio=round(pr, 2),
                n90=ncomp(.90), n95=ncomp(.95), n99=ncomp(.99),
                effective_dims=int(len(lam)))


# -----------------------------------------------------------------------------
# task-relevant DOF
# -----------------------------------------------------------------------------
def task_dof(X, Y, max_k=64, seed=0):
    """How many PCA components of X are needed to linearly predict Y to 95% of
    the achievable ceiling. Reports k95, the R^2 ceiling, and CCA correlations.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.cross_decomposition import CCA

    Xc = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xc = Xc - Xc.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    Z = U * S                                        # PCA scores
    K = int(min(max_k, Z.shape[1]))

    def cv_r2(k):
        scores = cross_val_score(Ridge(alpha=1.0), Z[:, :k], Y,
                                 cv=5, scoring="r2")
        return float(scores.mean())

    curve = [cv_r2(k) for k in range(1, K + 1)]
    ceil = max(curve)
    thresh = 0.95 * ceil if ceil > 0 else -np.inf
    k95 = next((i + 1 for i, r in enumerate(curve) if r >= thresh), K)

    # per-target ceiling at K components (Omega_m vs sigma_8 separately)
    per_target = [round(float(cross_val_score(Ridge(alpha=1.0), Z[:, :K],
                                              Y[:, j], cv=5, scoring="r2").mean()), 3)
                  for j in range(Y.shape[1])]

    ncca = int(min(Y.shape[1], Z.shape[1]))
    cca = CCA(n_components=ncca, max_iter=2000).fit(Xc, Y)
    Xs, Ys = cca.transform(Xc, Y)
    corrs = [round(float(np.corrcoef(Xs[:, i], Ys[:, i])[0, 1]), 3)
             for i in range(ncca)]

    return dict(k95=int(k95), r2_ceiling=round(ceil, 3),
                r2_per_target=per_target,
                cca_correlations=corrs,
                r2_curve={str(k): round(curve[k - 1], 3)
                          for k in (1, 2, 4, 8, 16, 32, K) if k <= K})


# -----------------------------------------------------------------------------
# feature builders
# -----------------------------------------------------------------------------
def coarse_features(maps, P):
    """Per-map mean over PxP blocks, flattened -> (n, (H//P)*(W//P)).

    This is a tokenizer-agnostic stand-in for the patch tokens: it is exactly
    what a mean-pooling linear tokenizer at patch size P would see, so its ID and
    task-DOF are properties of the data at that patch scale, not of any training.
    """
    n, H, W = maps.shape
    Hc, Wc = H // P, W // P
    m = maps[:, :Hc * P, :Wc * P].reshape(n, Hc, P, Wc, P).mean(axis=(2, 4))
    return m.reshape(n, Hc * Wc)


def sample_patches(maps, P, n_patches, seed=0):
    """Random PxP patches drawn across all maps -> (n_patches, P*P)."""
    rng = np.random.default_rng(seed)
    n, H, W = maps.shape
    idx = rng.integers(0, n, size=n_patches)
    yy = rng.integers(0, H - P + 1, size=n_patches)
    xx = rng.integers(0, W - P + 1, size=n_patches)
    out = np.empty((n_patches, P * P), dtype=np.float32)
    for t, (i, y, x) in enumerate(zip(idx, yy, xx)):
        out[t] = maps[i, y:y + P, x:x + P].reshape(-1)
    return out


def power_spectrum_features(maps, nbins=64):
    """Radially-averaged 2D power spectrum per map (log power), (n, nbins).

    This is the physically-correct cosmology summary and matches our pk foil
    (Ridge on log-pk reaches Omega_m R^2 ~ 0.83), so task-DOF measured on it is
    meaningful -- unlike a linear readout on pooled pixel means, which is blind
    to the spatial statistics cosmology actually lives in. Pass the LOG field:
    Mgas spans orders of magnitude, so a linear-field spectrum is dominated by a
    few bright pixels and the DC term rather than by clustering.
    """
    n, H, W = maps.shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    kk = np.sqrt(ky ** 2 + kx ** 2).ravel()
    bins = np.linspace(0, kk.max() + 1e-9, nbins + 1)
    which = np.clip(np.digitize(kk, bins) - 1, 0, nbins - 1)
    masks = [which == b for b in range(nbins)]
    out = np.empty((n, nbins), dtype=np.float64)
    chunk = 512                                          # cap the FFT footprint
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        F = np.fft.rfft2(maps[s:e], axes=(1, 2))
        Pf = (F.real ** 2 + F.imag ** 2).reshape(e - s, -1)
        for b, mask in enumerate(masks):
            out[s:e, b] = Pf[:, mask].mean(axis=1) if mask.any() else 0.0
    return np.log10(out + 1e-12)


# -----------------------------------------------------------------------------
# loading
# -----------------------------------------------------------------------------
def _npy_header(f):
    """(shape, fortran_order, dtype) for an open .npy, version-dispatched
    (numpy 2.x removed the private _read_array_header)."""
    major, minor = np.lib.format.read_magic(f)
    if (major, minor) == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if (major, minor) == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    raise ValueError(f"unsupported .npy version {major}.{minor}")


def _read_npy_rows(path, idx):
    """Read only rows `idx` (sorted) from a C-contiguous .npy, WITHOUT mmapping
    the whole file. mmap + fancy-index pulls the entire 3.9 GB file's pages into
    the cgroup page cache and OOM-kills an 8 GB container across repeated loads;
    seeking to the needed rows touches only ~subsample*row_bytes.
    """
    with open(path, "rb") as f:
        shape, fortran, dtype = _npy_header(f)
        if fortran:
            raise ValueError("Fortran-order .npy not supported")
        data_start = f.tell()
        row_shape = shape[1:]
        row_bytes = int(np.prod(row_shape)) * dtype.itemsize
        out = np.empty((len(idx),) + row_shape, dtype=dtype)
        for j, i in enumerate(idx):
            f.seek(data_start + int(i) * row_bytes)
            out[j] = np.frombuffer(f.read(row_bytes), dtype=dtype).reshape(row_shape)
    return out, shape[0]


def load_suite(map_spec, params_path, maps_per_sim, subsample, label_cols,
               do_log, seed=0):
    path, _, name = map_spec.partition(":")
    name = name or path.split("/")[-1]
    with open(path, "rb") as f:                       # header only
        shape, _fortran, _dtype = _npy_header(f)
    n_maps = shape[0]
    params = np.loadtxt(params_path)
    if params.ndim == 1:
        params = params[None, :]
    if n_maps % len(params) == 0 and n_maps != len(params):
        ratio = n_maps // len(params)
        if ratio != maps_per_sim:
            print(f"  [warn] {name}: {n_maps} maps / {len(params)} sims = "
                  f"{ratio}, not --maps-per-sim {maps_per_sim}; using {ratio}")
        labels = np.repeat(params, ratio, axis=0)
    elif len(params) == n_maps:
        labels = params
    else:
        sys.exit(f"  [error] {name}: cannot align {n_maps} maps to "
                 f"{len(params)} param rows")

    rng = np.random.default_rng(seed)
    take = min(subsample, n_maps)
    idx = np.sort(rng.choice(n_maps, size=take, replace=False))
    m, _ = _read_npy_rows(path, idx)
    m = m.astype(np.float32, copy=False)
    y = labels[idx][:, label_cols].astype(np.float64)

    if do_log:
        floor = m[m > 0].min() if (m > 0).any() else 1.0
        m = np.log10(np.clip(m, floor, None))

    # power-spectrum features on the (log) field -- log tames the Mgas dynamic
    # range so the spectrum reflects clustering, not a few bright pixels
    pk = power_spectrum_features(m)

    # sanity: Omega_m should sit in ~[0.1,0.5], sigma_8 in ~[0.6,1.0]
    lo, hi = y.min(0), y.max(0)
    print(f"  [{name}] {take} maps, label ranges: "
          + ", ".join(f"[{a:.3f},{b:.3f}]" for a, b in zip(lo, hi)))
    return name, m, y, pk


# -----------------------------------------------------------------------------
# suite-shift DOF
# -----------------------------------------------------------------------------
def suite_shift(feats_by_suite, labels_by_suite, P):
    """Alignment between the suite-discriminative direction and the
    cosmology-predictive directions at patch size P. High overlap => the encoder
    cannot separate 'which suite' from 'what cosmology' at this scale.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    names = list(feats_by_suite)
    if len(names) != 2:
        return None
    a, b = names
    Xa, Xb = feats_by_suite[a], feats_by_suite[b]
    ya, yb = labels_by_suite[a], labels_by_suite[b]

    X = np.vstack([Xa, Xb])
    suite = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]

    # CROSS-VALIDATED separability (scaler fit per-fold -> no train/test leakage).
    # A single train-set score memorises in 1024-dim (0.79 real vs 0.71 on SHUFFLED
    # labels -- the dof_tests control fired); CV is the honest number.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = float(cross_val_score(clf, X, suite, cv=cv, scoring="accuracy").mean())

    # cosmology-vs-suite direction alignment (descriptive geometry; full-data fit)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    w_suite = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, suite).coef_.ravel()
    w_suite /= (np.linalg.norm(w_suite) + 1e-12)
    Y = np.vstack([ya, yb])
    reg = Ridge(alpha=1.0).fit(Xs, Y)
    align = []
    for j in range(Y.shape[1]):
        w = reg.coef_[j] / (np.linalg.norm(reg.coef_[j]) + 1e-12)
        align.append(round(abs(float(w @ w_suite)), 3))

    return dict(suite_classifier_accuracy=round(acc, 3),
                cosine_suite_vs_label=align,
                note="suite_classifier_accuracy = 5-fold CV (held-out). cosine: "
                     "1.0 = suite dir == a label dir (worst), 0.0 = orthogonal.")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", required=True,
                    help="one or more path[:NAME] .npy map files")
    ap.add_argument("--params", nargs="+", required=True,
                    help="matching params_LH_*.txt (same order as --maps)")
    ap.add_argument("--patch-sizes", default="8,16,32")
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--n-patches", type=int, default=20000)
    ap.add_argument("--maps-per-sim", type=int, default=15)
    ap.add_argument("--label-cols", default="0,1",
                    help="param columns used as the readout target (Omega_m,sigma_8)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--mle-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0,
                    help="controls the map subsample and patch sampling")
    ap.add_argument("--out", default="dof_probe.json")
    args = ap.parse_args()

    if len(args.maps) != len(args.params):
        sys.exit("--maps and --params must have the same count / order")
    patch_sizes = [int(p) for p in args.patch_sizes.split(",")]
    label_cols = [int(c) for c in args.label_cols.split(",")]

    t0 = time.time()
    report = {"config": vars(args), "suites": {}}
    feats_by_suite, labels_by_suite = {}, {}

    for map_spec, params_path in zip(args.maps, args.params):
        name, m, y, pk = load_suite(map_spec, params_path, args.maps_per_sim,
                                    args.subsample, label_cols, not args.no_log,
                                    seed=args.seed)
        suite_rep = {"n_maps": int(len(m)), "patch_scales": {}}

        # data-manifold ID on a coarsened (P=4 -> 64x64) per-map cloud
        base = coarse_features(m, 4)
        suite_rep["data_manifold"] = {
            "twonn": round(twonn(base), 2),
            "mle": round(mle_id(base, args.mle_k), 2),
            **pca_stats(base),
            "feature_dims": int(base.shape[1]),
            "note": "per-map cloud at 64x64; ID ~ true DOF of a map",
        }

        # task-relevant DOF on the radially-averaged power spectrum (physical
        # cosmology summary; a property of the map+labels, not of patch size)
        suite_rep["task_relevant_dof"] = {"basis": "log power spectrum",
                                          "pk_bins": int(pk.shape[1]),
                                          **task_dof(pk, y)}

        for P in patch_sizes:
            patches = sample_patches(m, P, args.n_patches, seed=args.seed)
            feats = coarse_features(m, P)
            feats_by_suite.setdefault(name, {})[P] = feats
            labels_by_suite[name] = y
            suite_rep["patch_scales"][str(P)] = {
                "n_tokens_per_map": int((m.shape[1] // P) ** 2),
                "patch_cloud_id": {
                    "twonn": round(twonn(patches), 2),
                    "mle": round(mle_id(patches, args.mle_k), 2),
                },
                "feature_pca": pca_stats(feats),
            }

        td = suite_rep["task_relevant_dof"]
        p8 = suite_rep["patch_scales"].get("8", {}).get("patch_cloud_id", {}).get("twonn")
        print(f"  [{name}] data-ID(twoNN)={suite_rep['data_manifold']['twonn']}, "
              f"task k95={td['k95']}, pk-ceiling R2={td['r2_ceiling']}, "
              f"patch-ID@P8={p8}")

        report["suites"][name] = suite_rep

    # cross-suite shift, per patch size
    if len(report["suites"]) == 2:
        report["suite_shift"] = {}
        for P in patch_sizes:
            fb = {n: feats_by_suite[n][P] for n in feats_by_suite}
            lb = {n: labels_by_suite[n] for n in feats_by_suite}
            report["suite_shift"][str(P)] = suite_shift(fb, lb, P)

    report["elapsed_sec"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out} in {report['elapsed_sec']}s")


if __name__ == "__main__":
    main()
