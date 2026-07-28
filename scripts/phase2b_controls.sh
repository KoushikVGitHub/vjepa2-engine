#!/usr/bin/env bash
# PHASE 2b -- batch-96 confirmation + the full reviewer control suite (H1,H3,H4,H5,H7,H11).
# Resolves the two things Phase 2 left open: the batch-64 rank suppression, and whether the
# conv-stem win is the TOKENIZER (overlap/periodicity) or just a bigger/nonlinear tokenizer.
#
# NEEDS a 24-32GB 2-GPU pod: batch 48/GPU -> global 96 -> eff_rank ~35 (vs the ~21 at batch 64).
# On a 20GB card batch 48 OOMs -- this is why Phase 2 (0.766) is a conservative lower bound.
#
# TRAIN ARMS (global-batch 96, n-blocks=4, seed 0, 4000 steps, all matched -> clean A/B):
#   linear = honest baseline at batch 96          conv  = the Phase-2 winner, confirmed at full rank
#   mlp    = H1 param-matched disjoint-patch MLP   convz = H11 conv w/ zeros padding [add via ARMS]
# PROBE BATTERY (cheap, minutes each, paired one-per-GPU):
#   H4 multi-seed (3) on every arm      -> error bars on each R2
#   H7 random-init frozen probe         -> the "learned nothing" floor (trained - random = real learning)
#   H3 raw-tokenizer-stage probe        -> localises the gain to the tokenizer (conv vs linear)
#   H5 second field Mcdm                -> generality beyond Mgas
#
# LAUNCH: tmux new-session -d -s p2b "PEAK=<gpu_bf16_tflops> bash /workspace/vjepa2-engine/scripts/phase2b_controls.sh > /workspace/logs/p2b.log 2>&1"
set -uo pipefail
WS=/workspace; CKPT=$WS/checkpoints; LOG=$WS/logs; mkdir -p "$CKPT" "$LOG"; cd $WS/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=$WS/.cache/torch/kernels
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-4000}; SEED=${SEED:-0}; BATCH=${BATCH:-48}; PEAK=${PEAK:-312}; W=${W:-32}
ARMS=${ARMS:-"linear conv mlp"}     # append "convz" for the H11 zeros-padding ablation
SEEDS=${SEEDS:-"0 1 2"}
FIELD2=${FIELD2:-Mcdm}
BASE="--mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 --var-coef 5.0 --cov-coef 4e-2 --target-norm \
  --ckpt --peak-tflops $PEAK --d 1024 --layers 24 --heads 16 --img 256 --patch 8 --block 8 --n-blocks 4 \
  --steps $STEPS --batch $BATCH --save-every $STEPS --log-every 100"

# arm -> "train-flags|ckpt|probe-stem|probe-pad"
spec() { case "$1" in
  linear) echo "--stem linear|$CKPT/ckpt_p2b_linear.pt|linear|circular";;
  conv)   echo "--stem conv --stem-pad circular|$CKPT/ckpt_p2b_conv.pt|conv|circular";;
  mlp)    echo "--stem mlp|$CKPT/ckpt_p2b_mlp.pt|mlp|circular";;
  convz)  echo "--stem conv --stem-pad zeros|$CKPT/ckpt_p2b_convz.pt|conv|zeros";;
esac; }

echo "=== PHASE 2b: arms=[$ARMS] global-batch=$((BATCH*2)) steps=$STEPS seed=$SEED  $(date -u +%H:%M:%S) ==="
for A in $ARMS; do
  IFS='|' read -r FLAGS CK _ _ <<< "$(spec "$A")"
  echo "--- TRAIN $A (2-GPU) $(date -u +%H:%M:%S) ---"
  torchrun --standalone --nproc_per_node=2 src/train_fsdp.py $BASE $FLAGS --save "$CK" > "$LOG/p2b_train_$A.log" 2>&1
  echo "$A train rc=$? ($(grep -c '^step' "$LOG/p2b_train_$A.log" 2>/dev/null) step-logs)"
done

# probe one ckpt at all SEEDS on one GPU: $1 label $2 ckpt $3 stem $4 pad $5 field $6 stage $7 gpu $8 extra
probe_ms() {
  for s in $SEEDS; do
    CUDA_VISIBLE_DEVICES=$7 python scripts/run_probe.py --ckpt "$2" --field "$5" \
      --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 \
      --stem "$3" --stem-pad "$4" --probe-stage "$6" --no-atlas --seed "$s" --workers "$W" $8 \
      > "$LOG/p2b_${1}_s$s.log" 2>&1
  done
}
agg() {  # $1 label -> "label  Omega_m=a±b  sigma8=c±d"
  python - "$LOG" "$1" "$SEEDS" <<'PY'
import sys, re, statistics as st
log, lbl, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split()
om, s8 = [], []
for s in seeds:
    try: t = open(f"{log}/p2b_{lbl}_s{s}.log").read()
    except FileNotFoundError: continue
    m = re.search(r"R2\s*:\s*Omega_m=([0-9.]+)\s+sigma8=([0-9.]+)", t)
    if m: om.append(float(m[1])); s8.append(float(m[2]))
f = lambda v: (sum(v)/len(v), st.pstdev(v) if len(v) > 1 else 0.0)
if om:
    a,b = f(om); c,d = f(s8)
    print(f"{lbl:20s} Omega_m={a:.3f}±{b:.3f}  sigma8={c:.3f}±{d:.3f}  (n={len(om)})")
else:
    print(f"{lbl:20s} <no result>")
PY
}
# launch probe jobs 2-at-a-time (one per GPU) to avoid VRAM contention: pass "label|ckpt|stem|pad|field|stage|extra"
run_paired() {
  local i=0
  for job in "$@"; do
    IFS='|' read -r L CK S P F ST EX <<< "$job"
    probe_ms "$L" "$CK" "$S" "$P" "$F" "$ST" $((i%2)) "$EX" &
    i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
  done
  wait
}

echo "--- PROBE BATTERY $(date -u +%H:%M:%S) ---"
# H4: every trained arm, in-suite Mgas, encoder stage
JOBS=(); for A in $ARMS; do IFS='|' read -r _ CK S P <<< "$(spec "$A")"; JOBS+=("arm_$A|$CK|$S|$P|Mgas|encoder|"); done
run_paired "${JOBS[@]}"
# H7 random-init floor + H3 tokenizer-stage (conv,linear) + H5 second field (conv,linear)
run_paired \
  "H7_randinit|$CKPT/ckpt_p2b_conv.pt|conv|circular|Mgas|encoder|--random-init" \
  "H3_tok_conv|$CKPT/ckpt_p2b_conv.pt|conv|circular|Mgas|tokenizer|" \
  "H3_tok_linear|$CKPT/ckpt_p2b_linear.pt|linear|circular|Mgas|tokenizer|" \
  "H5_${FIELD2}_conv|$CKPT/ckpt_p2b_conv.pt|conv|circular|$FIELD2|encoder|" \
  "H5_${FIELD2}_linear|$CKPT/ckpt_p2b_linear.pt|linear|circular|$FIELD2|encoder|"

echo "=== PHASE 2b RESULTS (global-batch $((BATCH*2)), seeds [$SEEDS]) ==="
for A in $ARMS; do agg "arm_$A"; done
agg H7_randinit; agg H3_tok_conv; agg H3_tok_linear; agg "H5_${FIELD2}_conv"; agg "H5_${FIELD2}_linear"
echo "reference | linear@b64 0.546/0.372 | conv@b64 0.766/0.420 | pk-floor 0.818/0.331"
echo "reads | mlp≈linear & conv≫both ⇒ gain is OVERLAP not capacity (H1) | trained≫random ⇒ real learning not leakage (H7)"
echo "      | conv-tok≫linear-tok ⇒ gain sits in the tokenizer (H3) | conv≈convz ⇒ periodicity not load-bearing (H11)"
echo "=== PHASE 2b DONE ==="
