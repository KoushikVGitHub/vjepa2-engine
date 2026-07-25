#!/usr/bin/env bash
# Convergence run for the tokenizer A/B: extend both arms to STEPS, probe the checkpoint ladder
# -> R^2-vs-steps curve + the CONVERGED patch-16-vs-patch-8 verdict + the true gap to pk (0.818).
#
# 2-GPU data-parallel per arm. NCCL_P2P_DISABLE=1 works around the A4000 no-NVLink P2P deadlock
# (verified: 2-GPU steps clean with it). Benefits vs the single-GPU grad-accum A/B:
#   * both GPUs saturated + ~2x faster
#   * global batch = 32/GPU * 2 = 64, all-reduced in the SIGReg distributed path => a TRUE
#     64-sample batch statistic. This MATTERS: eff_rank <= #samples, so batch 32 caps rank ~12
#     (below the ~32-dim intrinsic cosmology dimension -> R^2 bottlenecked). Global 64 -> rank ~38
#     (>= 32), the regime where the keeper hit R^2 0.50. patch-8 peaks 15.24 GB/GPU here (fits 16 GB;
#     peak is on step 2 and deterministic, so no mid-run OOM).
# Arms run SEQUENTIALLY (each uses both cards); the probe ladder runs on BOTH GPUs (heavy patch-8
# probes split across GPU0/GPU1, light patch-16 probes fill GPU0).
#
#   tmux new-session -d -s cv "STEPS=4000 bash /workspace/ab_converge.sh > /workspace/cv.log 2>&1"
set -uo pipefail
WS=/workspace
cd /workspace/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=/workspace/.cache/torch/kernels
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-4000}
SAVE_EVERY=${SAVE_EVERY:-1000}
SEED=${SEED:-0}
PROBE_STEPS=${PROBE_STEPS:-"2000 $STEPS"}     # add 1000/3000 if the plateau is unclear
BATCH=${BATCH:-32}     # per-GPU; *2 GPUs = global 64 (>= 32-dim floor). patch-8 peaks 15.2GB/GPU.
C="--mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 --var-coef 5.0 --cov-coef 4e-2 --target-norm --ckpt --peak-tflops 77 --d 1024 --layers 24 --heads 16 --steps $STEPS --batch $BATCH --save-every $SAVE_EVERY --log-every 100"
echo "=== CONVERGENCE: STEPS=$STEPS global-batch=$((BATCH*2)) (2-GPU DP) save-every=$SAVE_EVERY probe@[$PROBE_STEPS] seed=$SEED ==="

echo "--- TRAIN patch16 (2-GPU) ---"
torchrun --standalone --nproc_per_node=2 src/train_fsdp.py $C --img 256 --patch 16 --block 4 --n-blocks 4 --save $WS/ckpt_p16.pt > $WS/cv_p16.log 2>&1
echo "patch16 train rc=$? ($(grep -c '^step' $WS/cv_p16.log 2>/dev/null) step-logs)"

echo "--- TRAIN patch8 (2-GPU) ---"
torchrun --standalone --nproc_per_node=2 src/train_fsdp.py $C --img 256 --patch 8 --block 8 --n-blocks 4 --save $WS/ckpt_p8.pt > $WS/cv_p8.log 2>&1
echo "patch8 train rc=$? ($(grep -c '^step' $WS/cv_p8.log 2>/dev/null) step-logs)"

# step_path inserts _step{N}; the final STEPS checkpoint lives at ckpt_pX.pt
ck() { if [ "$2" = "$STEPS" ]; then echo "$WS/ckpt_$1.pt"; else echo "$WS/ckpt_$1_step$2.pt"; fi; }
pr16() { CUDA_VISIBLE_DEVICES=$2 python scripts/run_probe.py --ckpt "$(ck p16 $1)" --field Mgas --img 256 --patch 16 --enc-d 1024 --enc-layers 24 --enc-heads 16 --no-atlas --seed $SEED > "$WS/pr_p16_$1.log" 2>&1; }
pr8()  { CUDA_VISIBLE_DEVICES=$2 python scripts/run_probe.py --ckpt "$(ck p8 $1)"  --field Mgas --img 256 --patch 8  --enc-d 1024 --enc-layers 24 --enc-heads 16 --no-atlas --seed $SEED > "$WS/pr_p8_$1.log"  2>&1; }

echo "--- PROBE ladder (both GPUs) ---"
(   # GPU0: final (heaviest) patch8 probe + all light patch16 probes
  pr8 "$STEPS" 0
  for S in $PROBE_STEPS; do pr16 "$S" 0; done
) &
(   # GPU1: the intermediate patch8 probe(s)
  for S in $PROBE_STEPS; do [ "$S" = "$STEPS" ] || pr8 "$S" 1; done
) &
wait
echo "probing done"

echo "=== CURVE (Mgas in-suite R2, seed=$SEED, 2-GPU global-batch 32) ==="
printf "%-8s | %-28s | %-28s\n" step "patch16 (Om / s8)" "patch8 (Om / s8)"
for S in $PROBE_STEPS; do
  R16=$(grep -A2 "IN-SUITE" "$WS/pr_p16_$S.log" 2>/dev/null | grep "R2" | grep -oE "Omega_m=[0-9.]+ +sigma8=[0-9.]+")
  R8=$(grep -A2 "IN-SUITE" "$WS/pr_p8_$S.log" 2>/dev/null | grep "R2" | grep -oE "Omega_m=[0-9.]+ +sigma8=[0-9.]+")
  printf "%-8s | %-28s | %-28s\n" "$S" "$R16" "$R8"
done
echo "pk-floor | Omega_m=0.818 sigma8=0.331 | Omega_m=0.818 sigma8=0.331"
echo "=== RESULT DONE ==="
