#!/usr/bin/env bash
# Harder-masking sweep on the WINNING patch-8 encoder (Track-3 step 2: the sigma8 lever).
#
# WHY: patch-8 already BEATS pk-alone on sigma8 (0.367 > 0.331) -> it captures non-Gaussian
# info a 2-point statistic misses. Harder masking removes more context, forcing the encoder to
# model longer-range / higher-order structure -> should push sigma8 further. This is INDEPENDENT
# of the rank ceiling (that caps Omega_m via the 32-dim floor; masking targets sigma8's non-Gaussian
# content, not representation rank), so it runs clean on the 16GB A4000 pod.
#
# SWEEP: mask ratio via --n-blocks at fixed --block 8 on the patch-8 (32x32=1024-token) grid:
#   n-blocks 4  = 4 x (8x8=64 patches) = 256/1024 ~= 25% masked  (== converged baseline, the control)
#   n-blocks 8  ~= 50% masked
#   n-blocks 12 ~= 75% masked
# (random blocks overlap, so effective ratio is a bit below nominal -- monotone is what matters.)
#
# COST: patch-8 plateaued by step 2000 in the convergence run (0.626->0.630 over 2000->4000), so
# STEPS=2000/arm is enough to read the sigma8 trend. 3 arms, 2-GPU each.
#
# Same frozen recipe + 2-GPU global-batch 64 (NCCL_P2P_DISABLE=1 for the A4000 no-NVLink P2P).
#   tmux new-session -d -s ms "bash /workspace/scripts/mask_sweep.sh > /workspace/logs/ms.log 2>&1"
set -uo pipefail
WS=/workspace
CKPT=$WS/checkpoints; LOG=$WS/logs; mkdir -p "$CKPT" "$LOG"   # organised layout (2026-07-26)
cd /workspace/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=/workspace/.cache/torch/kernels
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-2000}
SAVE_EVERY=${SAVE_EVERY:-2000}
SEED=${SEED:-0}
BATCH=${BATCH:-32}            # per-GPU; *2 GPUs = global 64 (>= 32-dim floor). patch-8 peaks ~15.2GB/GPU.
NBLOCKS=${NBLOCKS:-"4 8 12"}  # mask-ratio sweep (nominal ~25/50/75%)
C="--mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 --var-coef 5.0 --cov-coef 4e-2 --target-norm --ckpt --peak-tflops 77 --d 1024 --layers 24 --heads 16 --img 256 --patch 8 --block 8 --steps $STEPS --batch $BATCH --save-every $SAVE_EVERY --log-every 100"
echo "=== MASK SWEEP: patch-8 ViT-L, n-blocks in [$NBLOCKS], STEPS=$STEPS global-batch=$((BATCH*2)) seed=$SEED ==="

for NB in $NBLOCKS; do
  echo "--- TRAIN n-blocks=$NB (2-GPU) ---"
  torchrun --standalone --nproc_per_node=2 src/train_fsdp.py $C --n-blocks $NB --save $CKPT/ckpt_m$NB.pt > $LOG/ms_m$NB.log 2>&1
  echo "n-blocks=$NB train rc=$? ($(grep -c '^step' $WS/ms_m$NB.log 2>/dev/null) step-logs)"
done

echo "--- PROBE each arm (both GPUs) ---"
pr() { CUDA_VISIBLE_DEVICES=$2 python scripts/run_probe.py --ckpt $CKPT/ckpt_m$1.pt --field Mgas \
  --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 --no-atlas --seed $SEED > $LOG/pr_m$1.log 2>&1; }
# fan the (heavy patch-8) probes across the two GPUs: round-robin the arms
i=0
for NB in $NBLOCKS; do pr "$NB" $((i % 2)) & i=$((i+1)); done
wait
echo "probing done"

echo "=== CURVE (Mgas in-suite R2, patch-8, seed=$SEED, global-batch $((BATCH*2))) ==="
printf "%-14s | %-30s\n" "n-blocks (~%)" "Omega_m / sigma8"
for NB in $NBLOCKS; do
  R=$(grep -A2 "IN-SUITE" $LOG/pr_m$NB.log 2>/dev/null | grep "R2" | grep -oE "Omega_m=[0-9.]+ +sigma8=[0-9.]+")
  printf "%-14s | %-30s\n" "$NB" "$R"
done
echo "pk-alone floor | Omega_m=0.818 sigma8=0.331"
echo "=== RESULT DONE ==="
