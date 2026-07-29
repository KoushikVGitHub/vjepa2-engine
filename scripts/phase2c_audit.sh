#!/usr/bin/env bash
# PHASE 2c -- audit controls S3 (overlap isolation) + S2 (pretraining-seed error bar).
# Runs AFTER phase2b + h9 (needs both GPUs; ~3 train arms + probes). Matches phase2b BASE exactly.
#
# S3: convdisjoint at the SAME pretraining seed as phase2b's conv (train_fsdp default 1234) -> a
#     clean overlap contrast. conv >> convdisjoint => the Phase-2 gain is receptive-field OVERLAP;
#     conv ~= convdisjoint => it is depth/GroupNorm/nonlinearity, not overlap.
# S2: linear + conv at a SECOND pretraining seed (--seed 0) -> combined with phase2b's seed-1234
#     linear/conv gives a 2-seed spread on the linear-vs-conv delta. This is the error bar that
#     matters: phase2b's "multi-seed" only reseeded the PROBE HEAD (~+-0.03), not pretraining.
#
# NB phase2b arms pretrain at train_fsdp --seed DEFAULT=1234 (its BASE passes no --seed; the "seed 0"
#    comment in phase2b_controls.sh is WRONG). So convdisjoint here (no --seed) == phase2b's seed, and
#    the --seed 0 arms here are a genuinely different pretraining seed. Verified: grep default=1234.
set -uo pipefail
WS=/workspace; CKPT=$WS/checkpoints; LOG=$WS/logs; mkdir -p "$CKPT" "$LOG"; cd $WS/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=$WS/.cache/torch/kernels
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-4000}; BATCH=${BATCH:-48}; PEAK=${PEAK:-165}; W=${W:-32}
SEEDS=${SEEDS:-"0 1 2"}     # PROBE-HEAD seeds (per ckpt)
BASE="--mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 --var-coef 5.0 --cov-coef 4e-2 --target-norm \
  --ckpt --peak-tflops $PEAK --d 1024 --layers 24 --heads 16 --img 256 --patch 8 --block 8 --n-blocks 4 \
  --steps $STEPS --batch $BATCH --save-every $STEPS --log-every 100"

# label -> "train-flags|ckpt|probe-stem"  (probe pad is circular for all conv variants)
declare -A SPEC=(
  [convdisjoint]="--stem convdisjoint --stem-pad circular|$CKPT/ckpt_p2c_convdisjoint.pt|convdisjoint"
  [linear_s0]="--stem linear --seed 0|$CKPT/ckpt_p2c_linear_s0.pt|linear"
  [conv_s0]="--stem conv --stem-pad circular --seed 0|$CKPT/ckpt_p2c_conv_s0.pt|conv"
)
ARMS=${ARMS:-"convdisjoint linear_s0 conv_s0"}

echo "=== PHASE 2c: arms=[$ARMS] global-batch=$((BATCH*2)) steps=$STEPS  $(date -u +%H:%M:%S) ==="
for A in $ARMS; do
  IFS='|' read -r FLAGS CK _ <<< "${SPEC[$A]}"
  echo "--- TRAIN $A (2-GPU) $(date -u +%H:%M:%S) ---"
  torchrun --standalone --nproc_per_node=2 src/train_fsdp.py $BASE $FLAGS --save "$CK" > "$LOG/p2c_train_$A.log" 2>&1
  echo "$A train rc=$?"
done

# probe one ckpt at all head-SEEDS on one GPU: $1 label $2 ckpt $3 stem $4 gpu
probe_ms() {
  for s in $SEEDS; do
    CUDA_VISIBLE_DEVICES=$4 python scripts/run_probe.py --ckpt "$2" --field Mgas \
      --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 \
      --stem "$3" --stem-pad circular --probe-stage encoder --no-atlas --seed "$s" --workers "$W" \
      > "$LOG/p2c_${1}_s$s.log" 2>&1
  done
}
agg() {  # $1 label  $2 log-prefix (p2c or p2b_arm) -> mean +- headstd over SEEDS
  python - "$LOG" "$1" "$2" "$SEEDS" <<'PY'
import sys, re, statistics as st
log, lbl, pfx, seeds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
om, s8 = [], []
for s in seeds:
    try: t = open(f"{log}/{pfx}_{lbl}_s{s}.log").read()
    except FileNotFoundError: continue
    m = re.search(r"R2\s*:\s*Omega_m=(-?[0-9.]+)\s+sigma8=(-?[0-9.]+)", t)  # A3: keep negative R2 (H7 floor)
    if m: om.append(float(m[1])); s8.append(float(m[2]))
f = lambda v: (sum(v)/len(v), st.pstdev(v) if len(v) > 1 else 0.0)
if om:
    a,b = f(om); c,d = f(s8)
    print(f"{lbl:22s} Omega_m={a:.3f}+-{b:.3f}  sigma8={c:.3f}+-{d:.3f}  (n={len(om)} head-seeds)")
else:
    print(f"{lbl:22s} <no result>")
PY
}

echo "--- PROBE (in-suite Mgas, head-seeds [$SEEDS]) $(date -u +%H:%M:%S) ---"
i=0
for A in $ARMS; do
  IFS='|' read -r _ CK S <<< "${SPEC[$A]}"
  probe_ms "$A" "$CK" "$S" $((i%2)) &
  i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
done
wait

echo "=== PHASE 2c RESULTS (global-batch $((BATCH*2))) ==="
echo "# S3 overlap isolation (all seed 1234):"
agg convdisjoint p2c
agg conv         p2b_arm      # phase2b conv, seed 1234 (reference)
echo "# S2 pretraining-seed spread (linear & conv, seeds {1234 from phase2b, 0 here}):"
agg linear   p2b_arm          # linear seed 1234
agg linear_s0 p2c             # linear seed 0
agg conv     p2b_arm          # conv seed 1234
agg conv_s0  p2c              # conv seed 0
echo "reads | conv(1234) >> convdisjoint(1234) => gain is OVERLAP (S3) | conv(1234)~=conv(0) & linear(1234)~=linear(0) => the +0.22 delta is stable across pretraining seeds (S2)"
echo "fair pk floor (Mgas TEST, S1): Omega_m 0.834 / sigma8 0.446 ; pk+moments 0.837/0.544"
echo "=== PHASE 2c DONE ==="
