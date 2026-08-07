"""Deterministic gate statistics -- the product of the eval framework.

No LLM anywhere in here. Everything is a pure function of numbers, so it is unit
tested against known-answer cases (test_eval.py). Implements the reviewer's S2
must-fix: an upper-confidence-bounded run-to-run SD as the noise band (not
max-of-spreads), variance sources combined in quadrature, and a minimum
detectable effect reported on every comparison so the honest-null is
institutional rather than optional.
"""
import math

import numpy as np
from scipy import stats


def _z(p):
    return float(stats.norm.ppf(p))


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean. Screening only -- measures within-set
    sampling noise, NOT the seed/split variance a decisive claim must clear."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def sd_ucb(values, alpha=0.05):
    """95% UPPER confidence bound on the population SD from n samples (chi-square).

    At n=3 this is ~4.4x the point SD -- which is the point: 3 replicates cannot
    promise a tight band, and the gate must not pretend they can."""
    v = np.asarray(values, float)
    n = len(v)
    if n < 2:
        return float("inf")
    s = v.std(ddof=1)
    return float(s * math.sqrt((n - 1) / stats.chi2.ppf(alpha, n - 1)))


def combine_sd(*sds):
    """Independent variance sources ADD (quadrature); they do not max. Reviewer S2(b)."""
    finite = [s for s in sds if math.isfinite(s)]
    if len(finite) < len(sds):
        return float("inf")
    return float(math.sqrt(sum(s * s for s in finite)))


def mde(sigma, n, power=0.8, alpha=0.05):
    """Two-sample minimum detectable effect at equal n per arm."""
    if not math.isfinite(sigma) or n < 1:
        return float("inf")
    return float((_z(1 - alpha / 2) + _z(power)) * sigma * math.sqrt(2.0 / n))


def decide_comparison(arm_vals, floor_vals, required_replicates,
                      extra_sds=(), power=0.8, alpha=0.05):
    """Arm-vs-floor gate. Decision in {REFUSED, DECISIVE, NULL, INDECISIVE}.

    band = quadrature(UCB run-to-run SD of the arm, *extra axis SDs).
    An effect counts as DECISIVE only if it exceeds the band AND is at least the
    MDE (i.e. actually resolvable at this replicate count). Anything the design
    cannot resolve reads INDECISIVE, loudly -- reviewer S2.
    """
    arm = list(map(float, arm_vals))
    floor = list(map(float, floor_vals))
    if len(arm) < required_replicates or len(floor) < required_replicates:
        return dict(decision="REFUSED", n_arm=len(arm), n_floor=len(floor),
                    reasons=[f"replicates arm={len(arm)}/floor={len(floor)} "
                             f"< required {required_replicates}"])
    n = min(len(arm), len(floor))
    arm_m, floor_m = float(np.mean(arm)), float(np.mean(floor))
    delta = arm_m - floor_m
    band = combine_sd(sd_ucb(arm), *extra_sds)
    mde_val = mde(band, n, power, alpha)

    reasons = []
    if mde_val > abs(delta) or abs(delta) <= band:
        decision = "INDECISIVE"
        if mde_val > abs(delta):
            reasons.append(f"MDE {mde_val:.4f} > |delta| {abs(delta):.4f}")
        if abs(delta) <= band:
            reasons.append(f"|delta| {abs(delta):.4f} <= band {band:.4f}")
    elif delta > band:
        decision = "DECISIVE"
    else:  # delta < -band and resolvable
        decision = "NULL"
        reasons.append("arm below floor beyond the band (real no-improvement)")
    return dict(decision=decision, delta=delta, band=band, mde=mde_val, n=n,
                arm_mean=arm_m, floor_mean=floor_m, reasons=reasons)


def floor_parity(arm_cfg, floor_cfg, keys):
    """Reviewer S5: arm and floor must match on every listed key or the comparison
    is biased by unequal treatment. Returns the list of violations (empty = ok)."""
    return [f"{k}: arm={arm_cfg.get(k)!r} floor={floor_cfg.get(k)!r}"
            for k in keys if arm_cfg.get(k) != floor_cfg.get(k)]


def split_hygiene(prov, is_probe_split=False):
    """Reviewer S6. manifest_identity is required for any run (else position//15 !=
    sim and sim_split silently leaks). sim_disjoint is required for a probe split."""
    problems = []
    if not prov.get("manifest_identity", False):
        problems.append("manifest not identity -> position//15 != sim (leakage risk)")
    if is_probe_split and not prov.get("sim_disjoint", False):
        problems.append("probe split not asserted sim-disjoint")
    return problems
