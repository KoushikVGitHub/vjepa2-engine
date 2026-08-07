#!/usr/bin/env python3
"""
ridge_readout.py -- the READOUT CONTROL for probe / transfer claims.

The trained head is an attentive-pool MLP; on out-of-distribution features an
unregularised head produces meaningless magnitudes (a seed once swung to -7.96).
For any claim we also want the number a *ridge* on mean-pooled frozen features
gives -- the same regularisation the classical pk foil gets. This separates head
brittleness from encoder physics.

Two numbers per target (Omega_m, sigma8):
  * R2      -- ridge fit on the SOURCE train split, scored on the test split.
               For --heldout this is the RAW cross-suite R2 (fuses info + calibration).
  * r2aff   -- R2 after a per-target AFFINE refit (a*yhat + b) on the TARGET side.
               Removes a scale/offset the target never got to fit; reads as SKILL
               where raw R2 reads as deployability. (pk sigma8 foil: R2 -0.003, r2aff 0.228.)

Pure CPU: it reads the disk feature cache that run_probe.py already wrote (same
cache key via run_probe._feat_sig), so run the probe once to populate the cache,
then this refits with no encoder pass. Emits JSON for trace.log_run ingestion.

Usage (after a run_probe.py pass populated the cache):
  python scripts/ridge_readout.py --ckpt /workspace/checkpoints/ckpt_noise_s0.pt \
      --field Mgas --suite IllustrisTNG --enc-d 64 --enc-layers 4 --enc-heads 4 \
      --patch 8 --stem conv                       # in-suite ridge R2
  ... --heldout SIMBA                              # + cross-suite raw R2 and r2aff
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
# No sklearn: the GPU pods carry torch but not scikit-learn, and the network is
# often down (no pip). Ridge is a two-line closed form; we own it here.

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # scripts/  -> run_probe
sys.path.insert(0, os.path.join(HERE, "..", "src"))
# Reuse the EXACT cache-key builder + default geometry so we hit the same files
# run_probe wrote. Importing run_probe is side-effect-free (arg parsing is guarded).
from run_probe import _feat_sig, ENC           # noqa: E402

R2_TARGETS = ["Omega_m", "sigma8"]             # y[:,0], y[:,1]
ALPHAS = np.logspace(-3, 5, 25)                # ridge strength grid (picked on an internal split)


def r2_score(y, yhat):
    """Per-column R2 = 1 - SS_res/SS_tot, computed against each column's own mean."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)


def affine_refit_r2(y_true, y_pred):
    """r2aff: fit a*yhat + b per target on the TARGET side (least squares), score there.
    A representation can hold real signal (r2aff > 0) while raw R2 is negative purely from
    a scale/offset the target never fit. One dof per target (a, b) -- a cheap recalibration,
    not a new model."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = np.empty(y_true.shape[1])
    for j in range(y_true.shape[1]):
        A = np.vstack([y_pred[:, j], np.ones(len(y_pred))]).T
        (a, b), *_ = np.linalg.lstsq(A, y_true[:, j], rcond=None)
        out[j] = r2_score(y_true[:, j:j + 1], (a * y_pred[:, j] + b)[:, None])[0]
    return out


def enc_cfg_from_args(a):
    # MUST match run_probe's enc_cfg exactly (incl. stem_pad) or the geom-hash -- and thus
    # the cache file path -- diverges from the one run_probe wrote.
    return dict(img=a.img, patch=a.patch, d=a.enc_d, heads=a.enc_heads,
                layers=a.enc_layers, stem=a.stem, stem_pad=a.stem_pad)


def load_split(cdir, a, suite, split):
    """mean-pool the cached (N, n_patch, d) bf16 tokens -> (N, d) float32, plus raw y."""
    path = _feat_sig(cdir, a.ckpt, enc_cfg_from_args(a), a.field, suite, split,
                     a.probe_stage, a.random_init, a.seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no cache for {suite}/{split}:\n  {path}\n"
            f"Run run_probe.py first (same ckpt+geometry+field) to populate the cache, "
            f"or add the suite via --heldout {suite} there.")
    obj = torch.load(path, map_location="cpu")
    x = obj["x"].float()                       # (N, n_patch, d)
    x = x.mean(dim=1).numpy()                  # mean-pool -> (N, d)
    y = obj["y"].float().numpy()
    return x, y, path


class Ridge:
    """Standardize-then-ridge, numpy closed form. w = (Z'Z + aI)^-1 Z'y on standardized
    features (a NOT applied to the intercept). alpha is picked on an internal 80/20 row split
    of the training features by mean held-out R2, then the model is refit on all of train."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu_) / self.sd_
        n, d = Z.shape
        A = Z.T @ Z + self.alpha * np.eye(d)
        self.W_ = np.linalg.solve(A, Z.T @ (y - y.mean(0)))
        self.b_ = y.mean(0)
        return self

    def predict(self, X):
        return ((X - self.mu_) / self.sd_) @ self.W_ + self.b_


def fit_ridge(Xtr, ytr, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    cut = int(0.8 * len(Xtr))
    itr, iva = perm[:cut], perm[cut:]
    best_a, best_s = ALPHAS[0], -np.inf
    for a in ALPHAS:
        m = Ridge(a).fit(Xtr[itr], ytr[itr])
        s = np.nanmean(r2_score(ytr[iva], m.predict(Xtr[iva])))
        if s > best_s:
            best_s, best_a = s, a
    return Ridge(best_a).fit(Xtr, ytr)          # refit on full train at the chosen alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--field", default="Mgas")
    ap.add_argument("--suite", default="IllustrisTNG", help="source suite (ridge trained here)")
    ap.add_argument("--heldout", default=None, help="target suite for cross-suite R2 + r2aff")
    ap.add_argument("--img", type=int, default=ENC["img"])
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--enc-d", type=int, default=64)
    ap.add_argument("--enc-heads", type=int, default=4)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--stem", default="conv", choices=["linear", "conv", "convdisjoint", "mlp"])
    ap.add_argument("--stem-pad", default="circular", choices=["circular", "zeros"])
    ap.add_argument("--probe-stage", default="encoder", choices=["encoder", "tokenizer"])
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat-cache-dir", default=None)
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--out", default=None, help="write JSON verdict here")
    a = ap.parse_args()

    cdir = a.feat_cache_dir or os.path.join(a.data_root, "_probe_feat_cache")

    Xtr, ytr, ptr = load_split(cdir, a, a.suite, "train")
    Xte, yte, pte = load_split(cdir, a, a.suite, "test")
    reg = fit_ridge(Xtr, ytr)
    in_r2 = r2_score(yte, reg.predict(Xte))

    result = {
        "ckpt": os.path.abspath(a.ckpt), "field": a.field, "stem": a.stem,
        "enc": enc_cfg_from_args(a), "readout": "ridge(mean-pool)", "alpha": float(reg.alpha),
        "source_suite": a.suite,
        "in_suite_R2": {t: round(float(v), 4) for t, v in zip(R2_TARGETS, in_r2)},
        "n_train": len(Xtr), "n_test": len(Xte),
    }

    if a.heldout:
        # run_probe writes the cross-suite side under split name "heldout" (one full test
        # set; the head is trained on the SOURCE, so the target needs no train split).
        Xho, yho, pho = load_split(cdir, a, a.heldout, "heldout")
        yhat_ho = reg.predict(Xho)                   # SOURCE ridge -> TARGET features
        raw = r2_score(yho, yhat_ho)
        aff = affine_refit_r2(yho, yhat_ho)
        result["heldout_suite"] = a.heldout
        result["transfer_raw_R2"] = {t: round(float(v), 4) for t, v in zip(R2_TARGETS, raw)}
        result["transfer_r2aff"] = {t: round(float(v), 4) for t, v in zip(R2_TARGETS, aff)}
        result["n_heldout"] = len(Xho)

    print(json.dumps(result, indent=2))
    if a.out:
        tmp = f"{a.out}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, a.out)                       # atomic
        print(f"[ridge_readout] wrote {a.out}")


if __name__ == "__main__":
    main()
