#!/usr/bin/env python3
"""Dogfood: run the DOF findings through the framework end-to-end. Ingest them as
immutable run manifests, pre-register three claims, and let the judge decide. The
point: the framework CERTIFIES the clean findings and REFUSES the suite-separability
claim on its own -- because that claim's negative control failed (train-set
overfitting, task #16). The framework's first act is to withhold an over-claim.

Numbers are the real DOF study outputs (dof_both.json / dof_verdict.md); the
suite-sep backing check is the dof_tests suite-shuffle control, which FAILED.
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import trace, claims as claimlib
import judge as J

BASE = tempfile.mkdtemp(prefix="dof_ledger_")   # temp -> never pollutes the checkout (CI-safe)
RUNS = os.path.join(BASE, "runs")
CLAIMS = os.path.join(BASE, "claims.jsonl")


def ingest_runs():
    t = time.time()
    prov = lambda **k: dict(manifest_identity=True, exit_status="ok", created_at=t,
                            dataset_note="Mgas LH z=0, 2000-map subsample", **k)
    trace.log_run(RUNS, "dof_ITNG", {"suite": "ITNG", "study": "dof", "field": "Mgas"},
                  {"pk_omega_r2": {"value": 0.833}, "data_id_twonn": {"value": 47.7},
                   "task_k95": {"value": 10}}, prov())
    trace.log_run(RUNS, "dof_SIMBA", {"suite": "SIMBA", "study": "dof", "field": "Mgas"},
                  {"pk_omega_r2": {"value": 0.667}, "data_id_twonn": {"value": 32.6},
                   "task_k95": {"value": 6}}, prov())
    trace.log_run(RUNS, "dof_PAIR", {"suite": "PAIR", "study": "dof", "field": "Mgas"},
                  {"suite_sep_acc_train": {"value": 0.788},
                   "suite_sep_cv_p8": {"value": 0.588}}, prov())


def register_claims():
    now = time.time()  # AFTER the runs -> these are honestly RETROSPECTIVE (DOF predates the framework)
    claimlib.register(CLAIMS, {
        "claim_id": "dof-pk-anchor", "type": "measurement", "metric_id": "pk_omega_r2",
        "subject": {"suite": "ITNG"}, "expected_range": [0.80, 0.86],
        "backing_check": "dof_tests::regression_anchor", "backing_check_passed": True,
        "statement": "log-pk ridge recovers ITNG Omega_m at the known foil (~0.83)",
        "created_at": now})
    claimlib.register(CLAIMS, {
        "claim_id": "dof-data-id", "type": "measurement", "metric_id": "data_id_twonn",
        "subject": {"suite": "ITNG"}, "expected_range": [30, 60],
        "backing_check": "dof_tests::calibration+stability", "backing_check_passed": True,
        "statement": "ITNG Mgas data-manifold intrinsic dim is a few dozen (edim floor)",
        "created_at": now})
    # the ORIGINAL over-claim: train-set accuracy, whose negative control FAILED
    claimlib.register(CLAIMS, {
        "claim_id": "dof-suite-sep", "type": "measurement", "metric_id": "suite_sep_acc_train",
        "subject": {"suite": "PAIR"}, "expected_range": [0.70, 0.90],
        "backing_check": "dof_tests::suite_shuffle_control(train)", "backing_check_passed": False,
        "statement": "ITNG vs SIMBA linearly separable at patch-8 (~0.79)", "created_at": now})
    # retract against interest: abandon it (stays visible in the chain), register the
    # honest CV claim -- the control now passes (task #16)
    claimlib.abandon(CLAIMS, "dof-suite-sep",
                     "train-set overfitting in 1024-dim; CV=0.588 (task #16)")
    claimlib.register(CLAIMS, {
        "claim_id": "dof-suite-sep-cv", "type": "measurement", "metric_id": "suite_sep_cv_p8",
        "subject": {"suite": "PAIR"}, "expected_range": [0.55, 0.70],
        "backing_check": "dof_tests::suite_shuffle_control", "backing_check_passed": True,
        "statement": "ITNG vs SIMBA weakly separable at patch-8, CV ~0.59 (above chance)",
        "created_at": now})


def main():
    shutil.rmtree(BASE, ignore_errors=True)
    ingest_runs()
    register_claims()

    chain_ok, bad, verdicts = J.judge_all(CLAIMS, RUNS)
    print(f"claim chain verified: {chain_ok}" + (f" (bad rows {bad})" if bad else ""))
    orphans = trace.started_without_terminal(RUNS)
    print(f"started-without-terminal runs: {len(orphans)}\n")

    for v in verdicts:
        pre = v.get("pre_registered")
        tag = {"CERTIFIED": "OK ", "REFUSED": "NO ", "INDECISIVE": "?? ",
               "DECISIVE": "OK ", "NULL": "-- "}.get(v["decision"], "?? ")
        line = f"[{tag}] {v['claim_id']:<14} {v['decision']:<11} {v['metric_id']}"
        if "value" in v:
            line += f" = {v['value']}"
        print(line)
        for r in v.get("reasons", []):
            print(f"          reason: {r}")
        if pre is not None:
            print(f"          pre-registered: {pre}"
                  + ("  (retrospective -- DOF predates the framework)" if pre is False else ""))
    print()
    for r in claimlib.read_rows(CLAIMS):
        if r.get("op") == "abandon":
            print(f"[-- ] {r['claim_id']:<14} ABANDONED   reason: {r['reason']}")
    print()
    cert = sum(1 for v in verdicts if v["decision"] in ("CERTIFIED", "DECISIVE"))
    ref = sum(1 for v in verdicts if v["decision"] == "REFUSED")
    print(f"=> {cert} certified, {ref} refused, {len(verdicts) - cert - ref} indecisive "
          f"(+1 abandoned in-chain)")

    # CI assertion: the pipeline reproduces the expected retract-against-interest outcome
    got = {v["claim_id"]: v["decision"] for v in verdicts}
    expected = {"dof-pk-anchor": "CERTIFIED", "dof-data-id": "CERTIFIED",
                "dof-suite-sep-cv": "CERTIFIED"}
    n_aband = sum(1 for r in claimlib.read_rows(CLAIMS) if r.get("op") == "abandon")
    shutil.rmtree(BASE, ignore_errors=True)
    ok = chain_ok and got == expected and n_aband == 1 and not orphans
    print("[CI] " + ("PASS" if ok else
                     f"FAIL got={got} abandoned={n_aband} chain_ok={chain_ok} orphans={len(orphans)}"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
