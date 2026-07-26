#!/usr/bin/env bash
# Phase 0 (L14): RANK CONTROL / Omega_m disambiguation on 2x RTX 4090 (24 GB).
# patch-8 ViT-L, n-blocks 4, batch 64/GPU x2 = GLOBAL 128 -> rank ~72 (vs the global-64 run's rank ~25).
# Resolves the L12<->L13 contradiction: does Omega_m break past 0.63 once rank clears the 32-dim floor
# with real margin?
#   Omega_m > ~0.70 => the plateau was RANK-limited (L13 right; L12's "64 enough" was over-optimistic).
#   Omega_m ~0.63    => rank is NOT the bottleneck => tokenizer band-limiting => build conv-stem (Phase 2).
# This checkpoint is also the rank-matched CONTROL for Phase 1 (mask sweep) and Phase 2 (conv-stem).
#
#   tmux new-session -d -s p0 "bash /workspace/phase0_rank.sh > /workspace/p0.log 2>&1"
set -uo pipefail
WS=/workspace
CKPT=$WS/checkpoints; LOG=$WS/logs; mkdir -p "$CKPT" "$LOG"   # organised layout (2026-07-26)
cd /workspace/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=/workspace/.cache/torch/kernels
export NCCL_P2P_DISABLE=1                                   # 4090 pods have no NVLink either
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-4000}
SAVE_EVERY=${SAVE_EVERY:-1000}
SEED=${SEED:-0}
BATCH=${BATCH:-64}                                          # per-GPU; x2 = global 128 -> rank ~72
PROBE_STEPS=${PROBE_STEPS:-"2000 $STEPS"}
C="--mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 --var-coef 5.0 --cov-coef 4e-2 --target-norm --ckpt --peak-tflops 165 --d 1024 --layers 24 --heads 16 --img 256 --patch 8 --block 8 --n-blocks 4 --steps $STEPS --batch $BATCH --save-every $SAVE_EVERY --log-every 100"
echo "=== PHASE 0: patch-8 global-batch=$((BATCH*2)) STEPS=$STEPS seed=$SEED ==="

torchrun --standalone --nproc_per_node=2 src/train_fsdp.py $C --save $CKPT/ckpt_p8_b128.pt > $LOG/p0_train.log 2>&1
echo "train rc=$? ($(grep -c '^step' $LOG/p0_train.log 2>/dev/null) step-logs)"

ck() { if [ "$1" = "$STEPS" ]; then echo "$CKPT/ckpt_p8_b128.pt"; else echo "$CKPT/ckpt_p8_b128_step$1.pt"; fi; }
pr() { CUDA_VISIBLE_DEVICES=$2 python scripts/run_probe.py --ckpt "$(ck $1)" --field Mgas \
  --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 --no-atlas --seed $SEED > "$LOG/p0_pr_$1.log" 2>&1; }
echo "--- PROBE ladder (both GPUs) ---"
i=0; for S in $PROBE_STEPS; do pr "$S" $((i % 2)) & i=$((i+1)); done; wait
echo "probing done"

echo "=== CURVE (patch-8, global-128, Mgas in-suite R2, seed=$SEED) ==="
printf "%-8s | %-30s\n" step "Omega_m / sigma8"
for S in $PROBE_STEPS; do
  R=$(grep -A2 "IN-SUITE" "$LOG/p0_pr_$S.log" 2>/dev/null | grep "R2" | grep -oE "Omega_m=[0-9.]+ +sigma8=[0-9.]+")
  printf "%-8s | %-30s\n" "$S" "$R"
done
echo "ref: global-64 patch-8 = Omega_m 0.630 sigma8 0.367  |  pk-floor Omega_m 0.818 sigma8 0.331"
echo "=== RESULT DONE ==="
