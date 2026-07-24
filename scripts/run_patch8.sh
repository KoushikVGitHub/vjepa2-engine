#!/usr/bin/env bash
# Tokenizer A/B: does patch-16 -> patch-8 close the gap to the power spectrum (Omega_m R^2 0.818)?
#
# HYPOTHESIS: the patch-16 linear patch-embed AVERAGES AWAY sub-patch (high-k) power -- exactly
# where cosmology signal lives and exactly what P(k) sees for free -- so our SSL encoder sits
# BELOW the pk floor. If halving the patch to 8 lifts probe R^2 toward pk, the tokenizer was
# band-limiting. If it barely moves, the bottleneck is the SSL objective, not the tokenizer.
#
# CLEAN INTERNAL A/B (no historical confound): run BOTH arms here, identical except patch size +
# a mask compensation that holds PHYSICAL geometry fixed:
#   patch16 control : grid 16x16, --block 4 --n-blocks 4  -> 64 px blocks, ~25% target
#   patch8  test    : grid 32x32, --block 8 --n-blocks 4  -> 64 px blocks, ~25% target  (same!)
# Same loss recipe (the cov-decorrelation keeper), same batch, same steps. The DELTA is honest.
#
# Backbone stays ViT-L on BOTH arms on purpose -- isolate the tokenizer variable. Backbone
# downscale (ViT-S/B) and a conv/periodic stem are SEPARATE later arms (see learnings.md).
#
# Run detached on the pod so it survives SSH drops:
#   setsid nohup bash scripts/run_patch8.sh > /workspace/run_patch8.log 2>&1 &
set -euo pipefail

WS=/workspace
STEPS=${STEPS:-1000}      # comparable to the 0.493 keeper; bump once the delta is known
BATCH=${BATCH:-64}        # per-GPU; matched across arms. patch8 = 4x tokens -> --ckpt on both.
FIELD=${FIELD:-Mgas}

# --- shared loss recipe = the covariance-decorrelation keeper (README canonical, R^2 ~0.50) ---
RECIPE="--mode fsdp --bf16 --loss lejepa --sigreg-lambda 0.7 --lr 5e-5 \
        --var-coef 5.0 --cov-coef 4e-2 --target-norm --ckpt"

echo "==================== ARM A: patch-16 control ===================="
torchrun --standalone --nproc_per_node=2 src/train_fsdp.py \
  $RECIPE --img 256 --patch 16 --block 4 --n-blocks 4 \
  --d 1024 --layers 24 --heads 16 \
  --steps "$STEPS" --batch "$BATCH" --save "$WS/ckpt_p16.pt"

echo "==================== ARM B: patch-8 test ===================="
torchrun --standalone --nproc_per_node=2 src/train_fsdp.py \
  $RECIPE --img 256 --patch 8 --block 8 --n-blocks 4 \
  --d 1024 --layers 24 --heads 16 \
  --steps "$STEPS" --batch "$BATCH" --save "$WS/ckpt_p8.pt"

echo "==================== PROBE: patch-16 control ===================="
python scripts/run_probe.py --ckpt "$WS/ckpt_p16.pt" --field "$FIELD" \
  --img 256 --patch 16 --enc-d 1024 --enc-layers 24 --enc-heads 16 --no-atlas

echo "==================== PROBE: patch-8 test ===================="
python scripts/run_probe.py --ckpt "$WS/ckpt_p8.pt" --field "$FIELD" \
  --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 --no-atlas

echo "=== RESULT === compare IN-SUITE Omega_m/sigma8 R^2: patch-8 vs patch-16."
echo "patch-8 >> patch-16 (toward 0.818) => tokenizer was band-limiting; adopt small patch / conv stem."
echo "patch-8 ~= patch-16                => tokenizer is not the lever; bottleneck is the SSL objective."
