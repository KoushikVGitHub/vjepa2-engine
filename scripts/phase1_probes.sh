#!/usr/bin/env bash
# Phase-1 RESUME (probe-only). All 3 mask arms trained (ckpt_m4/m8/m12). A4000=Ampere -> stock torch.
# BOTTLENECK FIX: run_probe default --workers=4 starves the frozen-feature precompute (GPU idle at 0%).
# 128 cores here -> bump workers AND parallelize the two GPUs (loaders don't contend). Features are
# deterministic (shuffle off, augment off) so worker count changes speed only, not the sigma8 numbers.
set -uo pipefail
WS=/workspace; CKPT=$WS/checkpoints; LOG=$WS/logs; mkdir -p "$LOG"
cd $WS/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=$WS/.cache/torch/kernels
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=${SEED:-0}; W=${W:-32}
pr() { CUDA_VISIBLE_DEVICES=$1 python scripts/run_probe.py --ckpt $CKPT/ckpt_m$2.pt --field Mgas \
  --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 --no-atlas --seed $SEED --workers $3 \
  > $LOG/pr_m$2.log 2>&1; echo "m$2 rc=$?"; }
echo "--- WAVE 1: m4->GPU0 + m8->GPU1 (workers=$W each)  $(date -u +%H:%M:%S) ---"
pr 0 4 $W & pr 1 8 $W & wait
echo "--- WAVE 2: m12->GPU0 (workers=48)  $(date -u +%H:%M:%S) ---"
pr 0 12 48
echo "=== CURVE (Mgas in-suite R2, patch-8 ViT-L, global-batch 96, seed $SEED) ==="
printf '%-14s | %-30s\n' 'n-blocks (~%)' 'Omega_m / sigma8'
for NB in 4 8 12; do
  R=$(grep -A2 'IN-SUITE' $LOG/pr_m$NB.log 2>/dev/null | grep 'R2' | grep -oE 'Omega_m=[0-9.]+ +sigma8=[0-9.]+')
  printf '%-14s | %-30s\n' "$NB" "$R"
done
echo 'baseline m4 ref | Omega_m=0.630 sigma8=0.390   pk-floor | Omega_m=0.818 sigma8=0.331'
echo '=== PROBES DONE ==='
