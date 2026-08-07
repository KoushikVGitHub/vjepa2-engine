"""The decoupled judge -- deterministic, no LLM. Reads a pre-registered claim and
the run manifests it references, enforces provenance / floor-parity / split-hygiene
preconditions, and emits a structured verdict object. An LLM may later render prose
AROUND these fields but may not produce a number that is not here.

Verdict decisions:
  CERTIFIED   measurement in its pre-registered range with a passing backing check
  DECISIVE    comparison: arm beats floor beyond the band and the effect is resolvable
  NULL        comparison: arm is not better than floor beyond the band (honest null)
  INDECISIVE  effect not resolvable at this replicate count (MDE > effect), or within band
  REFUSED     a precondition failed (provenance, parity, replicate deficit, backing check)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import gate, trace, claims as claimlib


def _select(runs, selector):
    return [r for r in runs.values()
            if all(r["config"].get(k) == v for k, v in selector.items())]


def _pre_registered(claim, sel_runs):
    if not sel_runs:
        return None
    earliest = min(r["provenance"].get("created_at", float("inf")) for r in sel_runs)
    return claim.get("created_at", float("inf")) <= earliest


def _v(claim, decision, reasons, **extra):
    return dict(claim_id=claim.get("claim_id"), type=claim.get("type"),
                metric_id=claim.get("metric_id"), decision=decision,
                reasons=reasons, **extra)


def judge_measurement(claim, runs):
    sel = _select(runs, claim["subject"])
    if not sel:
        return _v(claim, "REFUSED", ["no run matches subject selector"])
    r = sel[0]
    prov = r["provenance"]
    hy = gate.split_hygiene(prov, is_probe_split=prov.get("is_probe_split", False))
    if hy:
        return _v(claim, "REFUSED", hy)
    if prov.get("exit_status", "ok") != "ok":
        return _v(claim, "REFUSED", [f"run exit_status={prov.get('exit_status')!r}"])
    if claim.get("backing_check_passed") is False:
        return _v(claim, "REFUSED",
                  [f"backing tool-check failed: {claim.get('backing_check')}"],
                  value=r["metrics"][claim["metric_id"]]["value"])
    val = r["metrics"][claim["metric_id"]]["value"]
    lo, hi = claim["expected_range"]
    pre = _pre_registered(claim, sel)
    if lo <= val <= hi:
        return _v(claim, "CERTIFIED", [], value=val, expected=[lo, hi],
                  pre_registered=pre, backing_check=claim.get("backing_check"))
    return _v(claim, "INDECISIVE",
              [f"value {val} outside pre-registered [{lo}, {hi}]"],
              value=val, expected=[lo, hi], pre_registered=pre)


def judge_comparison(claim, runs, runs_dir=None):
    arm = _select(runs, claim["arm"])
    floor = _select(runs, claim["floor"])
    if not arm or not floor:
        return _v(claim, "REFUSED",
                  [f"missing runs: arm={len(arm)} floor={len(floor)}"])
    # S6 split hygiene on every contributing run
    for r in arm + floor:
        hy = gate.split_hygiene(r["provenance"], is_probe_split=True)
        if hy:
            return _v(claim, "REFUSED", [f"{r['run_id']}: " + h for h in hy])
    # S5 floor parity
    viol = gate.floor_parity(arm[0]["config"], floor[0]["config"],
                             claim.get("floor_parity_keys", []))
    if viol:
        return _v(claim, "REFUSED", ["floor parity: " + v for v in viol])
    mid = claim["metric_id"]
    arm_vals = [r["metrics"][mid]["value"] for r in arm]
    floor_vals = [r["metrics"][mid]["value"] for r in floor]
    res = gate.decide_comparison(arm_vals, floor_vals,
                                 claim.get("required_replicates", 3),
                                 extra_sds=tuple(claim.get("extra_sds", ())))
    # S8: unfinished runs downgrade a DECISIVE verdict
    orphans = trace.started_without_terminal(runs_dir) if runs_dir else []
    if orphans and res["decision"] == "DECISIVE":
        res["decision"] = "INDECISIVE"
        res.setdefault("reasons", []).append(
            f"{len(orphans)} started-without-terminal runs pending explanation")
    return _v(claim, res.pop("decision"), res.pop("reasons", []), **res)


def judge_claim(claim, runs, runs_dir=None):
    if claim["type"] == "measurement":
        return judge_measurement(claim, runs)
    if claim["type"] == "comparison":
        return judge_comparison(claim, runs, runs_dir)
    return _v(claim, "REFUSED", [f"unknown claim type {claim['type']!r}"])


def judge_all(claims_path, runs_dir):
    """Verify the chain, then judge every active claim. Returns (chain_ok, verdicts)."""
    chain_ok, bad = claimlib.verify_chain(claims_path)
    runs = trace.read_runs(runs_dir)
    verdicts = [judge_claim(c, runs, runs_dir) for c in claimlib.active_claims(claims_path)]
    return chain_ok, bad, verdicts
