#!/usr/bin/env bash
# PHASE 2 -- conv-stem tokenizer vs linear patch-embed, controlled A/B for Omega_m.
#
# WHY: Phase 0/1 settled that Omega_m (a high-k / 2-point quantity) is capped at ~0.60 vs the
# pk-floor 0.818 because the LINEAR patch-8 embed box-averages each disjoint patch -> band-limits
# high spatial frequency. The conv-stem (--stem conv) uses OVERLAPPING stride-2 3x3 convs with
# CIRCULAR padding (CAMELS boxes are periodic), whose receptive fields span patch boundaries and
# preserve the small-scale power Omega_m needs. Masking is FIXED at n-blocks=4 (Phase-1 verdict:
# ratio is saturated, not a lever).
#
# CONTROLLED A/B: train BOTH stems at identical steps/batch/seed in one run so the ONLY variable is
# the tokenizer. The linear arm reproduces the patch-8 baseline (ref Omega_m 0.604, converged 0.630)
# at this exact global-batch; the conv arm is the test.
#   conv Omega_m rising toward 0.818  => tokenizer band-limiting CONFIRMED and FIXED (adopt for Phase 3)
#   conv Omega_m flat near ~0.60      => ceiling is capacity/data, not the tokenizer
#
# POD: 24-32GB 2-GPU (2x RTX 4090 / RTX PRO 4500). BATCH=48/GPU -> global 96 (rank-parity with Phase 0,
# eff_rank ~35). The 16GB A4000 canNOT train this at batch 48 -- it is a probe-only card for Phase 1.
# Conv-stem adds a small param/activation bump over linear; if OOM on a 24GB card, drop BATCH to 40
# (global 80) -- still above the ~batch>=64 rank-parity threshold.
#
# LAUNCH (detached, survives disconnect):
#   tmux new-session -d -s p2 "bash /workspace/scripts/phase2_convstem.sh > /workspace/logs/p2.log 2>&1"
#   tmux attach -t p2      # watch;  Ctrl-b d to detach
set -uo pipefail
WS=/workspace
CKPT=$WS/checkpoints; LOG=$WS/logs; mkdir -p "$CKPT" "$LOG"
cd $WS/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=$WS/.cache/torch/kernels
export NCCL_P2P_DISABLE=1                                  # no-NVLink 2-GPU (all these pods)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-4000}                                       # match the converged patch-8 run
SAVE_EVERY=${SAVE_EVERY:-4000}
SEED=${SEED:-0}
BATCH=${BATCH:-48}                                         # per-GPU; *2 = global 96. OOM on 24GB? -> BATCH=40.
PEAK=${PEAK:-165}                                          # bf16 peak TFLOPS: 165 (4090), 312 (A100/PRO-class)
STEMS=${STEMS:-"linear conv"}                              # A/B; set STEMS=conv to skip the baseline reproduction
W=${W:-32}                                                 # probe DataLoader workers (default 4 STARVES the GPU)
C="--mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 --var-coef 5.0 --cov-coef 4e-2 --target-norm \
   --ckpt --peak-tflops $PEAK --d 1024 --layers 24 --heads 16 --img 256 --patch 8 --block 8 --n-blocks 4 \
   --steps $STEPS --batch $BATCH --save-every $SAVE_EVERY --log-every 100"

echo "=== PHASE 2: conv-stem A/B  stems=[$STEMS] STEPS=$STEPS global-batch=$((BATCH*2)) seed=$SEED  $(date -u +%H:%M:%S) ==="
for S in $STEMS; do
  echo "--- TRAIN stem=$S (2-GPU) $(date -u +%H:%M:%S) ---"
  torchrun --standalone --nproc_per_node=2 src/train_fsdp.py $C --stem $S --save $CKPT/ckpt_stem_$S.pt \
    > $LOG/p2_train_$S.log 2>&1
  echo "stem=$S train rc=$? ($(grep -c '^step' $LOG/p2_train_$S.log 2>/dev/null) step-logs)"
done

echo "--- PROBE each stem (fan across both GPUs, workers=$W) $(date -u +%H:%M:%S) ---"
pr() { CUDA_VISIBLE_DEVICES=$2 python scripts/run_probe.py --ckpt $CKPT/ckpt_stem_$1.pt --field Mgas \
  --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 --stem $1 --no-atlas --seed $SEED \
  --workers $W > $LOG/p2_probe_$1.log 2>&1; echo "probe stem=$1 rc=$?"; }
i=0
for S in $STEMS; do pr "$S" $((i % 2)) & i=$((i+1)); done
wait

echo "=== CURVE (Mgas in-suite R2, patch-8 ViT-L, global-batch $((BATCH*2)), seed $SEED) ==="
printf "%-8s | %-30s\n" "stem" "Omega_m / sigma8"
for S in $STEMS; do
  R=$(grep -A2 "IN-SUITE" $LOG/p2_probe_$S.log 2>/dev/null | grep "R2" | grep -oE "Omega_m=[0-9.]+ +sigma8=[0-9.]+")
  printf "%-8s | %-30s\n" "$S" "$R"
done
echo "linear ref | Omega_m=0.604 (converged 0.630)     pk-floor | Omega_m=0.818 sigma8=0.331"
echo "=== PHASE 2 DONE ==="
