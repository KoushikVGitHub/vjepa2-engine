#!/usr/bin/env python3
"""Known-answer tests for the eval framework core. The gate is the product, so it
is tested the way dof_tests tests the estimators: against numbers computed by hand
or from theory, plus the L21 seed-swing trap it exists to catch. Exits non-zero on
any failure."""
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import gate, trace, claims as claimlib
import judge as J

R = []
def chk(name, ok, detail=""):
    R.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")


def test_gate_stats():
    print("\n== gate statistics (known-answer) ==")
    # sd_ucb at n=3 is ~4.41x the point SD (chi-square) -- the review's number.
    # std([0,1,2], ddof=1) == 1.0, so sd_ucb IS the inflation factor here.
    f = gate.sd_ucb([0.0, 1.0, 2.0])
    chk("sd_ucb n=3 factor ~4.41", abs(f - 4.415) < 0.05, f"factor={f:.3f}")
    fac = lambda v: gate.sd_ucb(v) / np.std(v, ddof=1)
    chk("UCB inflation factor shrinks with n", fac([0, 1, 2]) > fac(list(range(10))),
        f"n3={fac([0,1,2]):.2f} n10={fac(list(range(10))):.2f}")
    # quadrature, not max
    chk("combine_sd quadrature 0.3,0.4->0.5", abs(gate.combine_sd(0.3, 0.4) - 0.5) < 1e-9)
    # MDE matches the review table at sigma=0.115, n=3 (~0.26) and decreases with n
    chk("mde sigma=.115 n=3 ~0.26", abs(gate.mde(0.115, 3) - 0.263) < 0.01,
        f"{gate.mde(0.115,3):.3f}")
    chk("mde decreases with n", gate.mde(0.115, 10) < gate.mde(0.115, 3))


def test_decisions():
    print("\n== decision boundaries (incl. the L21 trap) ==")
    # REFUSED on replicate deficit
    chk("REFUSED < required replicates",
        gate.decide_comparison([0.8, 0.8], [0.72, 0.72], required_replicates=3)["decision"] == "REFUSED")
    # THE L21 TRAP: 3 tightly-clustered high seeds vs floor -- must NOT be DECISIVE at n=3
    trap = gate.decide_comparison([0.79, 0.80, 0.81], [0.72, 0.72, 0.72], required_replicates=3)
    chk("L21 trap: 0.79/0.80/0.81 vs 0.72 is INDECISIVE at n=3",
        trap["decision"] == "INDECISIVE", f"band={trap['band']:.3f} mde={trap['mde']:.3f}")
    # tiny delta within band -> INDECISIVE
    chk("tiny delta -> INDECISIVE",
        gate.decide_comparison([0.71, 0.73, 0.74], [0.72] * 3, 3)["decision"] == "INDECISIVE")
    # genuinely resolvable: tight arm, n=10, big gap -> DECISIVE
    arm = [0.80 + 0.008 * ((i % 3) - 1) for i in range(10)]     # ~0.80, spread ~0.008
    dec = gate.decide_comparison(arm, [0.72] * 10, required_replicates=3)
    chk("resolvable n=10 big gap -> DECISIVE", dec["decision"] == "DECISIVE",
        f"delta={dec['delta']:.3f} band={dec['band']:.3f} mde={dec['mde']:.3f}")
    # arm clearly worse -> NULL
    worse = gate.decide_comparison([0.60] * 10, [0.72] * 10, 3)
    chk("arm clearly worse -> NULL", worse["decision"] == "NULL", worse["decision"])


def test_parity_hygiene():
    print("\n== floor parity & split hygiene ==")
    chk("floor_parity flags mismatch",
        gate.floor_parity({"split_id": "A", "alpha_grid": 20}, {"split_id": "A", "alpha_grid": 1},
                          ["split_id", "alpha_grid"]) == ["alpha_grid: arm=20 floor=1"])
    chk("floor_parity clean when equal",
        gate.floor_parity({"split_id": "A"}, {"split_id": "A"}, ["split_id"]) == [])
    chk("split_hygiene flags non-identity manifest",
        gate.split_hygiene({"manifest_identity": False}) != [])
    chk("split_hygiene ok for identity+disjoint probe",
        gate.split_hygiene({"manifest_identity": True, "sim_disjoint": True}, is_probe_split=True) == [])


def test_chain_and_store():
    print("\n== hash-chained claims & immutable store ==")
    d = tempfile.mkdtemp()
    try:
        cp = os.path.join(d, "claims.jsonl")
        claimlib.register(cp, {"claim_id": "c1", "type": "measurement", "metric_id": "m"})
        claimlib.register(cp, {"claim_id": "c2", "type": "measurement", "metric_id": "m"})
        claimlib.abandon(cp, "c1", "superseded")
        ok, bad = claimlib.verify_chain(cp)
        chk("chain verifies", ok and not bad)
        chk("abandoned excluded from active",
            [c["claim_id"] for c in claimlib.active_claims(cp)] == ["c2"])
        # tamper a row -> chain breaks
        lines = open(cp).read().splitlines()
        lines[0] = lines[0].replace('"metric_id": "m"', '"metric_id": "TAMPERED"') \
                           .replace('"metric_id":"m"', '"metric_id":"TAMPERED"')
        open(cp, "w").write("\n".join(lines) + "\n")
        ok2, bad2 = claimlib.verify_chain(cp)
        chk("tamper detected", (not ok2) and bad2, f"bad rows {bad2}")

        # immutable run store
        rd = os.path.join(d, "runs")
        trace.log_run(rd, "r1", {"suite": "ITNG"}, {"m": {"value": 0.83}},
                      {"manifest_identity": True, "exit_status": "ok"})
        raised = False
        try:
            trace.log_run(rd, "r1", {"suite": "ITNG"}, {"m": {"value": 0.99}}, {})
        except FileExistsError:
            raised = True
        chk("run dir is immutable (re-log raises)", raised)
        chk("read_runs roundtrip", trace.read_runs(rd)["r1"]["metrics"]["m"]["value"] == 0.83)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    test_gate_stats()
    test_decisions()
    test_parity_hygiene()
    test_chain_and_store()
    nfail = sum(1 for _, ok in R if not ok)
    print("\n" + "=" * 56)
    print(f"SUMMARY: {len(R) - nfail}/{len(R)} passed, {nfail} failed")
    print("=" * 56)
    sys.exit(nfail)


if __name__ == "__main__":
    main()
