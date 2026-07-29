#!/usr/bin/env bash
# PHASE 2b -- H9 strict-hygiene control. Trains ONE conv arm with the probe's 100 test sims
# EXCLUDED from pretraining (--holdout-test-sims), across all 12 pooled fields, then probes it on
# exactly those held-out sims. Compare its R2 to the standard conv arm (ckpt_p2b_conv.pt):
#   heldout ~= standard  => in-suite unlabeled exposure is negligible; the tokenizer win is real
#                           even when the encoder has NEVER seen the test set.
#   heldout <  standard  => quantifies the in-suite optimism (the leakage the standard probe carries).
#
# Runs AFTER phase2b_controls.sh (needs both GPUs; ~2h train + minutes probe). Matches the main
# conv arm's BASE exactly, plus --holdout-test-sims, so the ONLY difference is the held-out sims.
# LAUNCH: tmux new-session -d -s h9 "PEAK=165 bash /workspace/vjepa2-engine/scripts/phase2b_h9.sh > /workspace/logs/h9.log 2>&1"
set -uo pipefail
WS=/workspace; CKPT=$WS/checkpoints; LOG=$WS/logs; mkdir -p "$CKPT" "$LOG"; cd $WS/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=$WS/.cache/torch/kernels
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-4000}; SEED=${SEED:-0}; BATCH=${BATCH:-48}; PEAK=${PEAK:-165}; W=${W:-32}
SEEDS=${SEEDS:-"0 1 2"}
CK=$CKPT/ckpt_p2b_conv_h9.pt
BASE="--mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 --var-coef 5.0 --cov-coef 4e-2 --target-norm \
  --ckpt --peak-tflops $PEAK --d 1024 --layers 24 --heads 16 --img 256 --patch 8 --block 8 --n-blocks 4 \
  --steps $STEPS --batch $BATCH --save-every $STEPS --log-every 100"

echo "=== PHASE 2b-H9: conv + holdout-test-sims, global-batch=$((BATCH*2)) steps=$STEPS  $(date -u +%H:%M:%S) ==="
echo "--- TRAIN conv_h9 (2-GPU, test sims held out of pretraining) $(date -u +%H:%M:%S) ---"
torchrun --standalone --nproc_per_node=2 src/train_fsdp.py $BASE \
  --stem conv --stem-pad circular --holdout-test-sims --save "$CK" > "$LOG/p2b_train_conv_h9.log" 2>&1
echo "conv_h9 train rc=$?  (holdout line: $(grep -m1 'H9 holdout' "$LOG/p2b_train_conv_h9.log"))"

echo "--- PROBE conv_h9 (in-suite Mgas, seeds [$SEEDS]) $(date -u +%H:%M:%S) ---"
for s in $SEEDS; do
  CUDA_VISIBLE_DEVICES=$(( (10#$s) % 2 )) python scripts/run_probe.py --ckpt "$CK" --field Mgas \
    --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 \
    --stem conv --stem-pad circular --probe-stage encoder --no-atlas --seed "$s" --workers "$W" \
    > "$LOG/p2b_H9_conv_s$s.log" 2>&1 &
  [ $(( (10#$s) % 2 )) -eq 1 ] && wait
done
wait

python - "$LOG" "$SEEDS" <<'PY'
import sys, re, statistics as st
log, seeds = sys.argv[1], sys.argv[2].split()
om, s8 = [], []
for s in seeds:
    try: t = open(f"{log}/p2b_H9_conv_s{s}.log").read()
    except FileNotFoundError: continue
    m = re.search(r"R2\s*:\s*Omega_m=(-?[0-9.]+)\s+sigma8=(-?[0-9.]+)", t)  # A3: keep negative R2 (H7 floor)
    if m: om.append(float(m[1])); s8.append(float(m[2]))
f = lambda v: (sum(v)/len(v), st.pstdev(v) if len(v) > 1 else 0.0)
if om:
    a,b = f(om); c,d = f(s8)
    print(f"H9_conv_heldout      Omega_m={a:.3f}±{b:.3f}  sigma8={c:.3f}±{d:.3f}  (n={len(om)})")
else:
    print("H9_conv_heldout      <no result>")
PY
echo "compare | conv-standard (ckpt_p2b_conv.pt, sees all sims unlabeled) vs conv-heldout above"
echo "read | heldout ~= standard => in-suite leakage negligible, tokenizer win survives strict hygiene (H9 closed)"
echo "=== PHASE 2b-H9 DONE ==="
