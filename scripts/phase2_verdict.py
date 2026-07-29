"""Phase-2 verdict aggregator: scan /workspace/logs for every probe arm, print the full R2 table, and
auto-resolve the pre-registered read rules (H1/H3/S3/confirmation/H7). Read-only; run on whichever pod
mounts the shared /workspace once conv/h9/phase2c have reported.

Parses ONLY the `R2 :` line (the IN-SUITE block prints RMSE / R2 / Coverage), anchored to the in-suite
block, negative-aware. Groups <prefix>_s{0,1,2}.log into mean +/- sample-sd.

  python scripts/phase2_verdict.py [--logs /workspace/logs]
"""
import argparse
import glob
import os
import re
import statistics as st

PK = {"Omega_m": 0.834, "sigma8": 0.446}  # S1-corrected fair pk floor (Mgas test)


def parse_r2(path):
    try:
        t = open(path).read()
    except OSError:
        return None
    m0 = re.search(r"=== IN-SUITE.*?(?==== HELD-OUT|\Z)", t, re.S)
    seg = m0.group(0) if m0 else t
    m = re.search(r"R2\s*:\s*Omega_m=(-?[0-9.]+)\s+sigma8=(-?[0-9.]+)", seg)
    return (float(m.group(1)), float(m.group(2))) if m else None


def agg(logs, prefix):
    """Aggregate <prefix>_s*.log -> (om_mean, om_sd, s8_mean, s8_sd, n) or None."""
    om, s8 = [], []
    for p in sorted(glob.glob(os.path.join(logs, f"{prefix}_s*.log"))):
        v = parse_r2(p)
        if v:
            om.append(v[0]); s8.append(v[1])
    if not om:
        return None
    f = lambda v: (sum(v) / len(v), st.stdev(v) if len(v) > 1 else 0.0)
    a, b = f(om); c, d = f(s8)
    return (a, b, c, d, len(om))


def row(label, r):
    if r is None:
        return f"  {label:26s} <no result>"
    a, b, c, d, n = r
    return f"  {label:26s} Omega_m={a:.3f}+-{b:.3f}   sigma8={c:.3f}+-{d:.3f}   (n={n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/workspace/logs")
    a = ap.parse_args()
    L = a.logs

    # (display label, log prefix). Two naming conventions exist:
    #   phase2b_probe_arms.sh -> p2b_arm_linear, p2b_H3tok_linear, p2b_H5Mcdm_linear
    #   phase2b_controls.sh   -> p2b_arm_conv,   p2b_H3_tok_conv,  p2b_H5_Mcdm_conv, p2b_H7_randinit
    #   phase2c_audit.sh      -> p2c_convdisjoint, p2c_linear_s0, p2c_conv_s0
    ENC = [
        ("linear (b96)", "p2b_arm_linear"),
        ("mlp (b96, disjoint)", "p2b_arm_mlp"),
        ("conv (b96)", "p2b_arm_conv"),
        ("convdisjoint (S3)", "p2c_convdisjoint"),
        ("H7 random-init floor", "p2b_H7_randinit"),
    ]
    TOK = [
        ("linear-tokenizer", "p2b_H3tok_linear"),
        ("mlp-tokenizer", "p2b_H3tok_mlp"),
        ("conv-tokenizer (H3)", "p2b_H3_tok_conv"),
        ("linear-tokenizer (ctrl)", "p2b_H3_tok_linear"),
    ]
    MCDM = [
        ("linear Mcdm", "p2b_H5Mcdm_linear"),
        ("mlp Mcdm", "p2b_H5Mcdm_mlp"),
        ("conv Mcdm (H5)", "p2b_H5_Mcdm_conv"),
    ]
    SEED = [
        ("linear seed1234", "p2b_arm_linear"),
        ("linear seed0 (S2)", "p2c_linear_s0"),
        ("conv seed1234", "p2b_arm_conv"),
        ("conv seed0 (S2)", "p2c_conv_s0"),
    ]
    H9 = [("conv heldout (H9)", "p2b_H9_conv"), ("conv standard", "p2b_arm_conv")]

    def val(prefix):
        r = agg(L, prefix)
        return r[0] if r else None  # Omega_m mean

    print("=" * 72)
    print("PHASE 2 VERDICT  (in-suite Mgas, batch 96, encoder stage unless noted)")
    print(f"fair pk floor (S1): Omega_m {PK['Omega_m']}  sigma8 {PK['sigma8']}   (pk+moments 0.837/0.544)")
    print("=" * 72)
    print("[encoder stage]")
    for lbl, p in ENC:
        print(row(lbl, agg(L, p)))
    print("[H3 raw tokenizer stage]")
    for lbl, p in TOK:
        print(row(lbl, agg(L, p)))
    print("[H5 second field Mcdm]")
    for lbl, p in MCDM:
        print(row(lbl, agg(L, p)))
    print("[S2 pretraining-seed spread]")
    for lbl, p in SEED:
        print(row(lbl, agg(L, p)))
    print("[H9 strict hygiene]")
    for lbl, p in H9:
        print(row(lbl, agg(L, p)))

    # -------- auto-resolved read rules (only when the needed arms exist) --------
    print("=" * 72)
    print("READ RULES (auto)")
    lin, mlp, conv, cdis, rnd = (val(p) for _, p in ENC)

    def verdict(cond, yes, no, need):
        if any(v is None for v in need):
            return "  [waiting on: " + ", ".join(n for n, v in zip(("linear","mlp","conv","convdisjoint","random"), need) if v is None) + "]"
        return "  " + (yes if cond else no)

    if conv is not None and mlp is not None and lin is not None:
        print("H1 capacity-vs-overlap:")
        print(f"    linear {lin:.3f} | mlp {mlp:.3f} | conv {conv:.3f}")
        if abs(conv - mlp) < 0.02:
            print("    => conv ~= mlp  ⇒  the tokenizer gain is CAPACITY/NONLINEARITY, not overlap.")
        elif conv > mlp:
            print(f"    => conv > mlp by {conv-mlp:+.3f}  ⇒  overlap adds on top of capacity (need S3 to confirm).")
        else:
            print("    => conv < mlp  ⇒  unexpected; mlp is the stronger disjoint tokenizer.")
    else:
        print("H1: waiting on conv" + ("" if mlp else " + mlp"))

    if conv is not None and cdis is not None:
        d = conv - cdis
        print("S3 overlap decider:")
        print(f"    conv {conv:.3f} | convdisjoint {cdis:.3f}  (Δ {d:+.3f})")
        print("    => " + ("conv ~= convdisjoint ⇒ OVERLAP is NOT the driver (depth/norm/capacity is)."
                           if abs(d) < 0.02 else
                           "conv > convdisjoint ⇒ receptive-field OVERLAP contributes."))
    else:
        print("S3: waiting on convdisjoint" + ("" if conv else " + conv"))

    if conv is not None and lin is not None:
        print(f"Confirmation (O_B96): conv {conv:.3f} vs linear@b96 {lin:.3f}  ⇒ real delta {conv-lin:+.3f} "
              f"(old +0.22 was vs linear@b64 0.546; linear rose to {lin:.3f} at b96, so part was rank).")

    ct, lt = val("p2b_H3_tok_conv"), (val("p2b_H3tok_linear") or val("p2b_H3_tok_linear"))
    if ct is not None and conv is not None:
        print(f"H3 tokenizer-vs-transformer: conv-encoder {conv:.3f} vs conv-tokenizer {ct:.3f} "
              f"⇒ {'transformer HELPS' if conv > ct else 'transformer LOSES info (raw tokens win)'}.")

    if rnd is not None and conv is not None:
        print(f"H7 leakage: conv {conv:.3f} − random {rnd:.3f} = {conv-rnd:+.3f} real learning "
              f"(random floor {'NEGATIVE' if rnd < 0 else 'positive'}).")

    h9h, h9s = val("p2b_H9_conv"), conv
    if h9h is not None and h9s is not None:
        print(f"H9 hygiene: heldout {h9h:.3f} vs standard {h9s:.3f} (Δ {h9h-h9s:+.3f}) "
              f"⇒ {'leakage negligible' if abs(h9h-h9s) < 0.03 else 'gap — but confounds 10% less data (one-sided)'}.")
    print("=" * 72)


if __name__ == "__main__":
    main()
