"""Pre-registered, hash-chained claims (reviewer S1 + S4).

A claim is registered BEFORE its runs are read -- it fixes the arm/floor selectors,
metric_id, expected range or decision rule, and required replicate count, so the
row set cannot be chosen after the numbers are seen. Claims are an append-only
JSONL file committed to git; git is the tamper-evident ledger, so we do not build
one. Each row carries prev_hash and row_hash = H(prev_hash || canonical(body)); a
mid-history edit breaks the chain and verify_chain finds it.

Editing a registered claim is impossible: you append a new claim (new id) and an
`abandon` row for the old one, with a reason. Abandoned claims stay visible --
invisible ones are the problem (reviewer S1, S10 multiplicity).
"""
import hashlib
import json
import os
import time


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _row_hash(prev_hash, body):
    return hashlib.sha256((prev_hash + _canon(body)).encode()).hexdigest()


def read_rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def _append(path, body):
    rows = read_rows(path)
    prev = rows[-1]["row_hash"] if rows else "GENESIS"
    body = {k: v for k, v in body.items() if k not in ("prev_hash", "row_hash")}
    body.setdefault("created_at", time.time())
    row = dict(body, prev_hash=prev, row_hash=_row_hash(prev, body))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def register(path, claim):
    """Append a claim row. `claim` must carry claim_id, type, metric_id, and the
    type-specific selectors/rule. Timestamp is set here (pre-registration time)."""
    assert claim.get("op", "register") == "register"
    for req in ("claim_id", "type", "metric_id"):
        if req not in claim:
            raise ValueError(f"claim missing required field {req!r}")
    return _append(path, dict(claim, op="register"))


def abandon(path, claim_id, reason):
    return _append(path, dict(op="abandon", claim_id=claim_id, reason=reason,
                              type="_meta", metric_id="_meta"))


def active_claims(path):
    """Latest-status view: registered claims whose id was not later abandoned."""
    rows = read_rows(path)
    abandoned = {r["claim_id"] for r in rows if r.get("op") == "abandon"}
    return [r for r in rows if r.get("op") == "register"
            and r["claim_id"] not in abandoned]


def verify_chain(path):
    """(ok, [bad_indices]). Recomputes every row_hash and checks prev linkage."""
    rows = read_rows(path)
    prev = "GENESIS"
    bad = []
    for i, row in enumerate(rows):
        body = {k: v for k, v in row.items() if k not in ("prev_hash", "row_hash")}
        if row.get("prev_hash") != prev or row.get("row_hash") != _row_hash(prev, body):
            bad.append(i)
        prev = row.get("row_hash")
    return (len(bad) == 0), bad
