#!/usr/bin/env python3
"""Pre-register the Astrid cross-suite transfer experiment BEFORE any run exists.

Fixes the arm/floor selectors, metric ids, replicate counts, and decision rules so
the row set cannot be chosen after the numbers are seen (reviewer S1/S4). Claims are
hash-chained; git is the tamper-evident ledger. Re-running is a no-op once the
claim_ids are present -- pre-registration must be append-once.

WHAT THE EXPERIMENT MUST LOG (via trace.log_run) for the judge to score these:
  config keys (selectors):
    repr        in {multisuite_declassify, multisuite_plain, singlesuite_ITNG,
                     raw_tokenizer, random_init}
    test_suite  = "Astrid" (held-out) | "IllustrisTNG" (in-suite retention)
    readout     = "ridge_r2aff" | "mlp"
    field       = "Mgas"
    seed        = pretraining seed (>=5 distinct for every comparison arm/floor)
  metrics: r2aff_omega_m, r2aff_sigma8, insuite_r2_omega_m, insuite_r2_sigma8,
           suite_cv_acc         (each {"value": float})
  provenance: manifest_identity=True, sim_disjoint=True, exit_status="ok", created_at

Success bars are set from TODAY's measured floors/bands (jepa_dof_arm_4090_campaign):
  raw-tok r2aff floor Om 0.130 / s8 0.195 ; lejepa arm r2aff sigma Om ~0.068 / s8 ~0.027 ;
  in-suite MLP band Om 0.866+-0.017 / s8 0.566+-0.025.
Comparisons decide by the gate band (UCB of the arm's >=5-seed SD), not a fixed
number -- the thresholds in the statements are documentation of intent, not the rule.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import claims as claimlib

CLAIMS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "claims", "astrid_transfer.jsonl")

PARITY = ["test_suite", "readout", "field"]   # arm & floor must match treatment (S5)

CLAIMS_TO_REGISTER = [
    # ---- HEADLINE: does the trained encoder beat the raw-tokenizer floor on the
    # held-out suite, on the calibration-corrected readout, by more than the band? ----
    dict(claim_id="astrid-r2aff-omega", type="comparison", metric_id="r2aff_omega_m",
         arm={"repr": "multisuite_declassify", "test_suite": "Astrid", "readout": "ridge_r2aff"},
         floor={"repr": "raw_tokenizer", "test_suite": "Astrid", "readout": "ridge_r2aff"},
         required_replicates=5, floor_parity_keys=PARITY,
         statement="multisuite+declassify beats raw-tok r2aff on held-out Astrid Omega_m "
                   "by >band (target ~0.24 vs floor 0.130)"),
    dict(claim_id="astrid-r2aff-sigma8", type="comparison", metric_id="r2aff_sigma8",
         arm={"repr": "multisuite_declassify", "test_suite": "Astrid", "readout": "ridge_r2aff"},
         floor={"repr": "raw_tokenizer", "test_suite": "Astrid", "readout": "ridge_r2aff"},
         required_replicates=5, floor_parity_keys=PARITY,
         statement="same for sigma8 (target ~0.25 vs floor 0.195)"),

    # ---- GUARDRAIL: the invariance must not destroy in-suite skill ----
    dict(claim_id="astrid-insuite-retention-omega", type="measurement",
         metric_id="insuite_r2_omega_m",
         subject={"repr": "multisuite_declassify", "test_suite": "IllustrisTNG", "readout": "mlp"},
         expected_range=[0.832, 1.0],   # plain-arm band lower bound 0.866-2*0.017
         backing_check="run_probe::in_suite_R2", backing_check_passed=None,
         statement="in-suite Omega_m stays within plain-arm band (>=0.832); below => invariance too aggressive"),
    dict(claim_id="astrid-insuite-retention-sigma8", type="measurement",
         metric_id="insuite_r2_sigma8",
         subject={"repr": "multisuite_declassify", "test_suite": "IllustrisTNG", "readout": "mlp"},
         expected_range=[0.516, 1.0],   # 0.566-2*0.025
         backing_check="run_probe::in_suite_R2", backing_check_passed=None,
         statement="in-suite sigma8 stays within plain-arm band (>=0.516)"),

    # ---- MECHANISTIC: de-classification must actually remove suite info ----
    dict(claim_id="astrid-suite-declassified", type="measurement", metric_id="suite_cv_acc",
         subject={"repr": "multisuite_declassify", "test_suite": "latent", "readout": "logistic_cv"},
         expected_range=[0.45, 0.55],   # chance +- tolerance (raw features were 0.588)
         backing_check="suite_probe::logistic_cv", backing_check_passed=None,
         statement="logistic suite-probe on the latent drops to chance ~0.5; else the loss "
                   "did not de-classify and the transfer number is not trustworthy"),

    # ---- ABLATIONS: attribute any gain to the right lever ----
    dict(claim_id="astrid-lever-declassify", type="comparison", metric_id="r2aff_omega_m",
         arm={"repr": "multisuite_declassify", "test_suite": "Astrid", "readout": "ridge_r2aff"},
         floor={"repr": "multisuite_plain", "test_suite": "Astrid", "readout": "ridge_r2aff"},
         required_replicates=5, floor_parity_keys=PARITY,
         statement="declassify loss adds transfer over multisuite-without-it (isolates lever 2)"),
    dict(claim_id="astrid-lever-multisuite", type="comparison", metric_id="r2aff_omega_m",
         arm={"repr": "multisuite_plain", "test_suite": "Astrid", "readout": "ridge_r2aff"},
         floor={"repr": "singlesuite_ITNG", "test_suite": "Astrid", "readout": "ridge_r2aff"},
         required_replicates=5, floor_parity_keys=PARITY,
         statement="multi-suite pretraining adds transfer over single-suite ITNG (isolates lever 1)"),
]


def main():
    existing = {r.get("claim_id") for r in claimlib.read_rows(CLAIMS)}
    now = time.time()
    newly = 0
    for c in CLAIMS_TO_REGISTER:
        if c["claim_id"] in existing:
            print(f"[skip] {c['claim_id']} already registered")
            continue
        claimlib.register(CLAIMS, dict(c, created_at=now))
        newly += 1
        print(f"[reg ] {c['claim_id']:<32} {c['type']:<11} {c['metric_id']}")

    ok, bad = claimlib.verify_chain(CLAIMS)
    active = claimlib.active_claims(CLAIMS)
    print(f"\nchain verified: {ok}" + (f"  BAD ROWS {bad}" if bad else ""))
    print(f"active pre-registered claims: {len(active)}  (+{newly} new this run)")
    print(f"ledger: {CLAIMS}")
    print("\nNext: run the Astrid arms/floors, trace.log_run each (>=5 seeds), then")
    print("  python eval/judge.py-style judge_all -> DECISIVE / NULL / INDECISIVE per claim.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
