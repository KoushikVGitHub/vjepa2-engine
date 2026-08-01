"""Phase-3: effective rank of each frozen representation, per suite.

Mechanism check for the tokenizer-beats-encoder result. If the pretrained transformer collapses
its pooled features into far fewer effective directions than its own tokenizer does, then "the
pretext discarded cosmology" has a concrete form: the surviving subspace is too small to carry
the label-relevant variance. Rank is measured on the SAME pooled features every readout used.

eff_rank = exp(entropy of the normalised eigenvalue spectrum of the feature covariance)
(participation-style effective rank; a flat spectrum over k dims gives exactly k).
"""
import os
import sys

import numpy as np

FEAT = "/dev/shm/feat"


def eff_rank(X, standardize=False):
    """standardize=True measures the CORRELATION spectrum instead of the covariance spectrum.

    The raw-covariance number is scale-dependent: a handful of high-variance dims makes it look
    rank-1 even when every dim is informative. Every readout here standardises its features, so
    the standardised rank is the one that describes what the ridge actually sees. Report both --
    the gap between them is itself the diagnostic (raw << standardised => the representation is
    dominated by a few large-amplitude directions).
    """
    Xc = X - X.mean(0)
    if standardize:
        sd = Xc.std(0)
        sd[sd < 1e-12] = 1.0
        Xc = Xc / sd
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    p = ev / ev.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    tags = sys.argv[1:] or ["conv_s0_encoder", "conv_s0_tokenizer",
                            "randinit_encoder", "randinit_tokenizer", "linear_s0_encoder"]
    print(f"{'representation':<28s} {'suite':<14s} {'rank_cov':>9s} {'rank_corr':>10s} {'dim':>6s}")
    print("-" * 70)
    for tag in tags:
        for suite in ["IllustrisTNG", "SIMBA"]:
            f = os.path.join(FEAT, f"{tag}_{suite}.npz")
            if not os.path.exists(f):
                continue
            X = np.load(f)["X"].astype(np.float64)
            print(f"{tag:<28s} {suite:<14s} {eff_rank(X):>9.1f} "
                  f"{eff_rank(X, standardize=True):>10.1f} {X.shape[1]:>6d}", flush=True)


if __name__ == "__main__":
    main()
