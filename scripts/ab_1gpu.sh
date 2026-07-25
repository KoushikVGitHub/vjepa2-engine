#!/usr/bin/env bash
# Parallel single-GPU A/B orchestrator (patch-16 control vs patch-8 test).
#
# WHY single-GPU-parallel instead of run_patch8.sh's 2-GPU FSDP:
# the 2x RTX A4000 pod has NO NVLink, so FSDP's PCIe P2P NCCL collectives DEADLOCK
# on the first all-gather (symptom: 100% GPU util but only ~55% TDP power draw, and
# step-0 never logs). Single-GPU (world_size=1) makes NCCL collectives no-ops, so it
# runs clean. Running the two arms on the two GPUs concurrently recovers the wall-time.
#
# Both arms are IDENTICAL except patch size + a mask compensation holding physical
# geometry fixed (patch16: block4/n4 = 64px @25%; patch8: block8/n4 = 64px @25%).
# Batch is matched across arms (patch-8's 1024 tokens OOM a 16GB A4000 above batch ~16,
# and the SIGReg loss scales with token count -> BATCH=16 is the safe matched fit).
#
# Launch detached in tmux (survives SSH drops):
#   tmux new-session -d -s ab "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     STEPS=1000 BATCH=16 bash /workspace/ab_1gpu.sh > /workspace/ab_full.log 2>&1"
set -uo pipefail
WS=/workspace
cd /workspace/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=/workspace/.cache/torch/kernels
STEPS=${STEPS:-1000}
BATCH=${BATCH:-16}
C="--mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 --var-coef 5.0 --cov-coef 4e-2 --target-norm --ckpt --peak-tflops 77 --d 1024 --layers 24 --heads 16 --steps $STEPS --batch $BATCH --log-every 50"
echo "=== A/B launch: STEPS=$STEPS BATCH=$BATCH ==="

# distinct rdzv ports so the two torchruns don't collide on the same host
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29511 --rdzv-id=p16 \
  src/train_fsdp.py $C --img 256 --patch 16 --block 4 --n-blocks 4 --save $WS/ckpt_p16.pt > $WS/arm_p16.log 2>&1 &
P16=$!
CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29512 --rdzv-id=p8 \
  src/train_fsdp.py $C --img 256 --patch 8 --block 8 --n-blocks 4 --save $WS/ckpt_p8.pt > $WS/arm_p8.log 2>&1 &
P8=$!
echo "P16=$P16 P8=$P8"
wait $P16; echo "ARM patch16 exit rc=$?"
wait $P8;  echo "ARM patch8  exit rc=$?"

echo "=== PROBE patch16 (GPU0) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/run_probe.py --ckpt $WS/ckpt_p16.pt --field Mgas \
  --img 256 --patch 16 --enc-d 1024 --enc-layers 24 --enc-heads 16 --no-atlas > $WS/probe_p16.log 2>&1; echo "probe16 rc=$?"
echo "=== PROBE patch8 (GPU1) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/run_probe.py --ckpt $WS/ckpt_p8.pt --field Mgas \
  --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 --no-atlas > $WS/probe_p8.log 2>&1; echo "probe8 rc=$?"

echo "=== RESULT ==="
echo "--- patch16 IN-SUITE ---"; grep -A3 "IN-SUITE" $WS/probe_p16.log
echo "--- patch8 IN-SUITE ---";  grep -A3 "IN-SUITE" $WS/probe_p8.log
