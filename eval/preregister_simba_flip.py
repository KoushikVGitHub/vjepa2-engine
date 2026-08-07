#!/usr/bin/env python3
"""Re-pre-registration after the quick signal: the Astrid held-out target proved trivially easy
(floors ~0.7 r2aff vs SIMBA's ~0.15), so it cannot discriminate de-classification's benefit. We
FLIP to hold out the HARD suite (SIMBA), training on ITNG+Astrid.

This appends to the SAME hash-chained ledger: it ABANDONS the 7 Astrid claims (with reason, they
stay visible -- honesty about the design change) and REGISTERS the SIMBA-held-out set. The chain is
the tamper-evident record; the abandoned rows document that we changed the target BEFORE seeing the
flipped run's numbers (only the n=2 Astrid signal, which motivated the flip, was seen).

Run only AFTER confirming ITNG-vs-Astrid separability > chance (else the adversary has no work).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import claims as claimlib

CLAIMS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "claims", "astrid_transfer.jsonl")
PARITY = ["test_suite", "readout", "field"]

ABANDON_REASON = ("Astrid held-out target is trivially easy (n=2: floors ~0.68-0.74 r2aff vs "
                  "SIMBA's ~0.13-0.20); AND ITNG-vs-Astrid separability is 0.512 (~chance) so "
                  "suite-de-classification of that pair is a no-op. Pivoted to FEEDBACK-parameter "
                  "invariance (shed A_SN/A_AGN, known per-map, varies within ITNG), train ITNG only, "
                  "test truly-unseen SIMBA -- the real confound, SIMBA genuinely held out.")

OLD = ["astrid-r2aff-omega", "astrid-r2aff-sigma8", "astrid-insuite-retention-omega",
       "astrid-insuite-retention-sigma8", "astrid-suite-declassified",
       "astrid-lever-declassify", "astrid-lever-multisuite"]

NEW = [
    dict(claim_id="simba-r2aff-omega", type="comparison", metric_id="r2aff_omega_m",
         arm={"repr": "feedback_invariance", "test_suite": "SIMBA", "readout": "ridge_r2aff"},
         floor={"repr": "raw_tokenizer", "test_suite": "SIMBA", "readout": "ridge_r2aff"},
         required_replicates=5, floor_parity_keys=PARITY,
         statement="feedback-invariant encoder (train ITNG) beats raw-tok r2aff on HARD held-out "
                   "SIMBA Omega_m by >band (floor ~0.13)"),
    dict(claim_id="simba-r2aff-sigma8", type="comparison", metric_id="r2aff_sigma8",
         arm={"repr": "feedback_invariance", "test_suite": "SIMBA", "readout": "ridge_r2aff"},
         floor={"repr": "raw_tokenizer", "test_suite": "SIMBA", "readout": "ridge_r2aff"},
         required_replicates=5, floor_parity_keys=PARITY,
         statement="same for sigma8 (floor ~0.195)"),
    dict(claim_id="simba-insuite-retention-omega", type="measurement", metric_id="insuite_r2_omega_m",
         subject={"repr": "feedback_invariance", "test_suite": "IllustrisTNG", "readout": "mlp"},
         expected_range=[0.832, 1.0], backing_check="run_probe::in_suite_R2", backing_check_passed=None,
         statement="in-suite Omega_m stays within plain-arm band (>=0.832); feedback-shedding must "
                   "not gut cosmology (feedback is varied independently of cosmology in LH)"),
    dict(claim_id="simba-insuite-retention-sigma8", type="measurement", metric_id="insuite_r2_sigma8",
         subject={"repr": "feedback_invariance", "test_suite": "IllustrisTNG", "readout": "mlp"},
         expected_range=[0.516, 1.0], backing_check="run_probe::in_suite_R2", backing_check_passed=None,
         statement="in-suite sigma8 stays within plain-arm band (>=0.516)"),
    dict(claim_id="simba-feedback-shed", type="measurement", metric_id="feedback_r2",
         subject={"repr": "feedback_invariance", "test_suite": "latent", "readout": "adversary_r2"},
         expected_range=[-0.15, 0.15], backing_check="train::feedback_r2_final", backing_check_passed=None,
         statement="the adversary's feedback-R2 on the final latent is ~0 (feedback no longer "
                   "decodable); else the GRL did not shed feedback and the transfer number isn't trusted"),
    dict(claim_id="simba-lever-feedback", type="comparison", metric_id="r2aff_omega_m",
         arm={"repr": "feedback_invariance", "test_suite": "SIMBA", "readout": "ridge_r2aff"},
         floor={"repr": "feedback_plain", "test_suite": "SIMBA", "readout": "ridge_r2aff"},
         required_replicates=5, floor_parity_keys=PARITY,
         statement="feedback invariance (lambda=1) adds SIMBA transfer over the identical lambda=0 arm "
                   "(isolates the de-confounding lever from the ITNG-only corpus)"),
]


def main():
    existing = {r.get("claim_id") for r in claimlib.read_rows(CLAIMS)}
    abandoned = {r["claim_id"] for r in claimlib.read_rows(CLAIMS) if r.get("op") == "abandon"}
    for cid in OLD:
        if cid in existing and cid not in abandoned:
            claimlib.abandon(CLAIMS, cid, ABANDON_REASON)
            print(f"[abandon] {cid}")
    now = time.time()
    for c in NEW:
        if c["claim_id"] in existing:
            print(f"[skip] {c['claim_id']} already registered")
            continue
        claimlib.register(CLAIMS, dict(c, created_at=now))
        print(f"[reg ] {c['claim_id']:<32} {c['type']:<11} {c['metric_id']}")
    ok, bad = claimlib.verify_chain(CLAIMS)
    active = claimlib.active_claims(CLAIMS)
    print(f"\nchain verified: {ok}" + (f"  BAD {bad}" if bad else ""))
    print(f"active claims: {len(active)} (Astrid set abandoned but visible in chain)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
