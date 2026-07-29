#!/usr/bin/env bash
# Probe-only battery for ALREADY-TRAINED arms (NO training). Splits the phase2b battery so the light
# arms (linear, mlp) can be probed on a 24GB pod while the conv arm -- which OOMs at batch 48 on 24GB
# (CONTEXT_GRAPH.md decision 19) -- is trained+probed on a 32GB+ pod at the SAME batch 48 (matched).
#
# Per arm: H4 encoder-stage Mgas over head-seeds (error bar) + H3 raw-tokenizer-stage Mgas
# (localises where any gain sits) + H5 second field Mcdm (generality). Arms run concurrently, one GPU
# each; seeds within an arm are sequential.
#
# A3 FIX: the aggregator is negative-R2-aware (-?) AND anchored to the IN-SUITE block, so a negative
# floor (e.g. H7 random-init) is not silently dropped and a future SIMBA block cannot be misread.
#
# LAUNCH: tmux new-session -d -s pr "W=32 ARMS='linear mlp' bash scripts/phase2b_probe_arms.sh > /workspace/logs/probe_linmlp.log 2>&1"
set -uo pipefail
WS=/workspace; CKPT=$WS/checkpoints; LOG=$WS/logs; mkdir -p "$LOG"; cd $WS/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=$WS/.cache/torch/kernels
W=${W:-32}; SEEDS=${SEEDS:-"0 1 2"}; FIELD2=${FIELD2:-Mcdm}
declare -A SPEC=(                      # arm -> "ckpt|stem"
  [linear]="$CKPT/ckpt_p2b_linear.pt|linear"
  [mlp]="$CKPT/ckpt_p2b_mlp.pt|mlp"
  [conv]="$CKPT/ckpt_p2b_conv.pt|conv"
)
ARMS=${ARMS:-"linear mlp"}

pf() {  # $1 label $2 ckpt $3 stem $4 field $5 stage $6 seed $7 gpu $8 extra
  CUDA_VISIBLE_DEVICES=$7 python scripts/run_probe.py --ckpt "$2" --field "$4" \
    --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 \
    --stem "$3" --stem-pad circular --probe-stage "$5" --no-atlas --seed "$6" --workers "$W" $8 \
    > "$LOG/p2b_${1}_s$6.log" 2>&1
}
agg() {  # $1 label -> mean+-samplestd over SEEDS (A3-safe: neg-aware + IN-SUITE-anchored)
  python - "$LOG" "$1" "$SEEDS" <<'PY'
import sys, re, statistics as st
log, lbl, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split()
om, s8 = [], []
for s in seeds:
    try: t = open(f"{log}/p2b_{lbl}_s{s}.log").read()
    except FileNotFoundError: continue
    m0 = re.search(r"=== IN-SUITE.*?(?==== HELD-OUT|\Z)", t, re.S)   # anchor to in-suite block
    seg = m0.group(0) if m0 else t
    m = re.search(r"R2\s*:\s*Omega_m=(-?[0-9.]+)\s+sigma8=(-?[0-9.]+)", seg)  # R2 line ONLY (block prints RMSE/R2/Coverage); -? keeps neg R2
    if m: om.append(float(m.group(1))); s8.append(float(m.group(2)))
f = lambda v: (sum(v)/len(v), st.stdev(v) if len(v) > 1 else 0.0)
if om:
    a,b = f(om); c,d = f(s8)
    print(f"{lbl:24s} Omega_m={a:.3f}+-{b:.3f}  sigma8={c:.3f}+-{d:.3f}  (n={len(om)} head-seeds)")
else:
    print(f"{lbl:24s} <no result>")
PY
}

run_arm() {  # $1 arm $2 gpu
  IFS='|' read -r CK S <<< "${SPEC[$1]}"
  for s in $SEEDS; do pf "arm_$1" "$CK" "$S" Mgas encoder "$s" "$2" ""; done   # H4 encoder Mgas
  pf "H3tok_$1"      "$CK" "$S" Mgas    tokenizer 0 "$2" ""                     # H3 tokenizer stage
  pf "H5${FIELD2}_$1" "$CK" "$S" "$FIELD2" encoder 0 "$2" ""                    # H5 second field
}

echo "=== PROBE [$ARMS] (no training; conv deferred to 32GB+ pod per CONTEXT_GRAPH decision 19)  $(date -u +%H:%M:%S) ==="
i=0
for A in $ARMS; do run_arm "$A" $((i%2)) & i=$((i+1)); [ $((i%2)) -eq 0 ] && wait; done
wait

echo "=== RESULTS (in-suite Mgas, encoder stage, seeds [$SEEDS]) ==="
for A in $ARMS; do agg "arm_$A"; done
echo "# H3 raw-tokenizer stage (Mgas):"; for A in $ARMS; do agg "H3tok_$A"; done
echo "# H5 $FIELD2 (encoder):";          for A in $ARMS; do agg "H5${FIELD2}_$A"; done
echo "fair pk floor (Mgas TEST, S1): Omega_m 0.834 / sigma8 0.446"
echo "read | mlp ~= linear (both disjoint) = the capacity floor; conv (32GB+ pod) must beat BOTH ⇒ overlap not capacity (H1/S3)"
echo "=== PROBE [$ARMS] DONE ==="
