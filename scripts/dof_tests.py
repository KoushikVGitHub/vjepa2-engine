#!/usr/bin/env python3
"""
dof_tests.py -- reproducible confirmation harness for the DOF grounding study.

Locks down every finding in dof_verdict.md with the control that would break it.
Imports the SAME estimator code that produced the findings (dof_probe.py) -- a
reimplementation would prove nothing. Prints a PASS/FAIL table and exits non-zero
on any failure.

  1. CALIBRATION   synthetic manifolds of KNOWN dim -> TwoNN/MLE recover them,
                   and we quantify the estimator bias (so ID~40 is interpretable).
  2. ANCHOR        ITNG pk Omega_m R^2 lands at the known foil (0.83).
  3. CONTROLS      shuffle labels -> pk task-DOF collapses to ~0;
                   shuffle suite  -> separability falls to ~0.5. Signal is real.
  4. STABILITY     vary seed & subsample -> ID/k95/anchor/suite stable (mean+-std).
  5. DETERMINISM   same seed -> byte-identical arrays and metrics.

Usage (on the pod, with the volume mounted):
  python3.13 dof_tests.py \
      --maps "/workspace/data/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy:ITNG" \
             "/workspace/data/Maps_Mgas_SIMBA_LH_z=0.00.npy:SIMBA" \
      --params /workspace/data/params_LH_IllustrisTNG.txt \
               /workspace/data/params_LH_SIMBA.txt
Run with no data args to execute only the (data-free) calibration test.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import numpy as np

import dof_probe as D
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "dof_probe.py")


RESULTS = []  # (name, passed, detail)


def record(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


# -----------------------------------------------------------------------------
# 1. estimator calibration on known-dimension synthetic manifolds
# -----------------------------------------------------------------------------
def test_calibration():
    print("\n== 1. estimator calibration (synthetic known dim) ==")
    n, ambient = 3000, 256
    rng = np.random.default_rng(0)
    rows, ok_monotone = [], True
    prev = -1
    for d in (1, 5, 10, 20, 40):
        Z = rng.standard_normal((n, d))
        A = rng.standard_normal((d, ambient))          # linear embed preserves dim
        X = Z @ A
        tw, ml = D.twonn(X), D.mle_id(X, 10)
        rows.append((d, tw, ml))
        # each estimate should be in a sane band around d and increase with d
        in_band = (0.5 * d - 1) <= tw <= (1.5 * d + 2)
        ok_monotone &= tw > prev
        prev = tw
        record(f"calib d={d}", in_band,
               f"TwoNN={tw:.1f} MLE={ml:.1f} (true {d})")
    # report the bias so ID~40 is interpretable
    bias = np.mean([(tw - d) / d for d, tw, _ in rows if d >= 10])
    record("calib monotonic", ok_monotone,
           f"TwoNN increases with true dim; mean rel-bias(d>=10)={bias:+.2f}")


# -----------------------------------------------------------------------------
# data helpers
# -----------------------------------------------------------------------------
def load_both(args, subsample, seed):
    out = {}
    for spec, pp in zip(args.maps, args.params):
        name, m, y, pk = D.load_suite(spec, pp, args.maps_per_sim, subsample,
                                      [0, 1], True, seed=seed)
        out[name] = dict(m=m, y=y, pk=pk)
    return out


def suite_summary(s):
    base = D.coarse_features(s["m"], 4)
    td = D.task_dof(s["pk"], s["y"])
    return dict(data_id=D.twonn(base), k95=td["k95"],
                omega=td["r2_per_target"][0], sigma=td["r2_per_target"][1],
                ceiling=td["r2_ceiling"])


def suite_sep(data, P=8):
    names = list(data)
    feats = {n: D.coarse_features(data[n]["m"], P) for n in names}
    labels = {n: data[n]["y"] for n in names}
    return D.suite_shift(feats, labels, P)


def run_probe(args, seed, subsample, patch="8"):
    """Run dof_probe.py in a fresh process (full memory reclaim -> no OOM ratchet
    in the 8 GB cgroup) and return its parsed JSON. This is also the honest
    reproducibility unit: metrics come from re-executing the artifact generator."""
    out = f"/tmp/dof_run_s{seed}_n{subsample}.json"
    cmd = ["python3.13", PROBE, "--maps", *args.maps, "--params", *args.params,
           "--patch-sizes", patch, "--subsample", str(subsample),
           "--n-patches", "8000", "--seed", str(seed), "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dof_probe failed (seed={seed} n={subsample}):\n{r.stderr[-500:]}")
    with open(out) as f:
        return json.load(f)


def stab_metrics(j, P="8"):
    itng = j["suites"]["ITNG"]
    ss = j["suite_shift"][P]
    return dict(data_id=itng["data_manifold"]["twonn"],
                k95=itng["task_relevant_dof"]["k95"],
                omega=itng["task_relevant_dof"]["r2_per_target"][0],
                suite_acc=ss["suite_classifier_accuracy"],
                suite_cos=max(ss["cosine_suite_vs_label"]))


# -----------------------------------------------------------------------------
# 2. regression anchor
# -----------------------------------------------------------------------------
def test_anchor(data):
    print("\n== 2. regression anchor (pipeline locked to known pk foil) ==")
    s = suite_summary(data["ITNG"])
    record("ITNG pk Omega_m R2 == foil 0.83", 0.80 <= s["omega"] <= 0.86,
           f"{s['omega']:.3f} (expect [0.80,0.86])")


# -----------------------------------------------------------------------------
# 3. negative controls
# -----------------------------------------------------------------------------
def test_controls(data, seed):
    print("\n== 3. negative controls (signal is real, not overfitting) ==")
    rng = np.random.default_rng(seed)

    # 3a. label shuffle -> task-DOF ceiling collapses
    for name in data:
        s = data[name]
        real = D.task_dof(s["pk"], s["y"])["r2_ceiling"]
        yp = s["y"][rng.permutation(len(s["y"]))]
        perm = D.task_dof(s["pk"], yp)["r2_ceiling"]
        record(f"{name} label-shuffle kills task-DOF",
               real >= 0.45 and perm < 0.05,
               f"real={real:.3f}  shuffled={perm:.3f}")

    # 3b. suite shuffle -> CV separability falls to chance (both real & shuffled
    # under CV; the train-set version of this control is what fired originally)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    real = suite_sep(data, 8)["suite_classifier_accuracy"]          # CV real
    names = list(data)
    fa = D.coarse_features(data[names[0]]["m"], 8)
    fb = D.coarse_features(data[names[1]]["m"], 8)
    X = np.vstack([fa, fb])
    suite = np.r_[np.zeros(len(fa)), np.ones(len(fb))]
    suite_perm = suite[rng.permutation(len(suite))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc_perm = float(cross_val_score(clf, X, suite_perm, cv=cv, scoring="accuracy").mean())
    record("suite-shuffle -> chance; real CV genuinely above it",
           0.42 <= acc_perm <= 0.58 and real >= acc_perm + 0.05,
           f"real(CV)={real:.3f}  shuffled(CV)={acc_perm:.3f}")


# -----------------------------------------------------------------------------
# 4. stability across seed & subsample (the noise band)
# -----------------------------------------------------------------------------
def test_stability(args):
    print("\n== 4. stability (seed & subsample, isolated subprocesses) ==")
    ms = [stab_metrics(run_probe(args, sd, 2000)) for sd in (0, 1, 2)]
    idv = np.array([m["data_id"] for m in ms])
    k95v = [m["k95"] for m in ms]
    omv = np.array([m["omega"] for m in ms])
    accv = np.array([m["suite_acc"] for m in ms])
    cosv = [m["suite_cos"] for m in ms]
    record("ITNG data-ID stable over seeds", idv.std() / idv.mean() < 0.12,
           f"{idv.mean():.1f}+-{idv.std():.1f} (CV {idv.std()/idv.mean():.2f})")
    record("ITNG task k95 stable", (max(k95v) - min(k95v)) <= 3,
           f"k95={k95v} (range<=3)")
    record("ITNG Omega_m R2 stable", omv.std() < 0.03,
           f"{omv.mean():.3f}+-{omv.std():.3f}")
    record("suite CV acc stable & above chance", accv.std() < 0.05 and accv.mean() > 0.55,
           f"{accv.mean():.3f}+-{accv.std():.3f} (5-fold CV, chance=0.5)")
    record("suite~cosmology orthogonal (all seeds)", max(cosv) < 0.30,
           f"max cosine over seeds = {max(cosv):.3f} (<0.30)")

    ids = np.array([stab_metrics(run_probe(args, 0, ss))["data_id"]
                    for ss in (1000, 2000, 4000)])
    record("ITNG data-ID stable over subsample", ids.std() / ids.mean() < 0.15,
           f"n=1k/2k/4k -> {ids.round(1)} (CV {ids.std()/ids.mean():.2f})")


# -----------------------------------------------------------------------------
# 5. determinism
# -----------------------------------------------------------------------------
def test_determinism(args):
    print("\n== 5. determinism (same seed -> identical) ==")
    a, b = stab_metrics(run_probe(args, 0, 2000)), stab_metrics(run_probe(args, 0, 2000))
    diffs = {k: abs(a[k] - b[k]) for k in a}
    record("dof_probe reproducible at fixed seed", max(diffs.values()) < 1e-9,
           f"max metric |delta| = {max(diffs.values()):.1e}")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+")
    ap.add_argument("--params", nargs="+")
    ap.add_argument("--maps-per-sim", type=int, default=15)
    args = ap.parse_args()

    test_calibration()
    if args.maps and args.params:
        assert len(args.maps) == len(args.params)
        data0 = load_both(args, 2000, 0)
        test_anchor(data0)
        test_controls(data0, 0)
        del data0; gc.collect()          # free before subprocess sweeps (shared 8GB cgroup)
        test_stability(args)
        test_determinism(args)
    else:
        print("\n(no --maps/--params: ran calibration only)")

    n_fail = sum(1 for _, p, _ in RESULTS if not p)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} passed, {n_fail} failed")
    print("=" * 60)
    sys.exit(n_fail)


if __name__ == "__main__":
    main()
