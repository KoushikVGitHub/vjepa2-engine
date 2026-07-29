#!/usr/bin/env bash
# H4 control -- probe ONE checkpoint at several seeds, report mean±std R2 (an error bar on the
# point estimate, since probe head-init noise is ~±0.03). Reusable helper; also used by phase2b.
#
# Usage:
#   CKPT=/workspace/checkpoints/ckpt_stem_conv.pt STEM=conv FIELD=Mgas SEEDS="0 1 2" GPU=0 \
#     bash scripts/probe_multiseed.sh
# Optional: PAD=circular|zeros  STAGE=encoder|tokenizer  EXTRA="--random-init"  W=32
set -uo pipefail
WS=/workspace; LOG=$WS/logs; mkdir -p "$LOG"; cd $WS/vjepa2-engine
export PYTORCH_KERNEL_CACHE_PATH=$WS/.cache/torch/kernels
CKPT=${CKPT:?set CKPT=/path/to/ckpt.pt}
STEM=${STEM:-conv}; FIELD=${FIELD:-Mgas}; SEEDS=${SEEDS:-"0 1 2"}
GPU=${GPU:-0}; PAD=${PAD:-circular}; STAGE=${STAGE:-encoder}; W=${W:-32}; EXTRA=${EXTRA:-}
tag=$(basename "$CKPT" .pt)_${FIELD}_${STAGE}$( [ -n "$EXTRA" ] && echo _rand )

echo "=== multiseed probe: $(basename "$CKPT") stem=$STEM pad=$PAD field=$FIELD stage=$STAGE seeds=[$SEEDS] ==="
for s in $SEEDS; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_probe.py --ckpt "$CKPT" --field "$FIELD" \
    --img 256 --patch 8 --enc-d 1024 --enc-layers 24 --enc-heads 16 \
    --stem "$STEM" --stem-pad "$PAD" --probe-stage "$STAGE" --no-atlas --seed "$s" --workers "$W" $EXTRA \
    > "$LOG/ms_${tag}_s$s.log" 2>&1
  R=$(grep -A2 "IN-SUITE" "$LOG/ms_${tag}_s$s.log" | grep R2 | grep -oE "Omega_m=[0-9.]+ +sigma8=[0-9.]+")
  echo "  seed $s: ${R:-<no result, see $LOG/ms_${tag}_s$s.log>}"
done

python - "$LOG" "$tag" "$SEEDS" <<'PY'
import sys, re, statistics as st
log, tag, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split()
om, s8 = [], []
for s in seeds:
    try:
        t = open(f"{log}/ms_{tag}_s{s}.log").read()
    except FileNotFoundError:
        continue
    m = re.search(r"R2\s*:\s*Omega_m=(-?[0-9.]+)\s+sigma8=(-?[0-9.]+)", t)  # A3: keep negative R2 (H7 floor)
    if m:
        om.append(float(m[1])); s8.append(float(m[2]))
f = lambda v: (sum(v) / len(v), st.pstdev(v) if len(v) > 1 else 0.0)
if om:
    a, b = f(om); c, d = f(s8)
    print(f"=== {tag}: Omega_m {a:.3f} ± {b:.3f}   sigma8 {c:.3f} ± {d:.3f}   (n={len(om)}) ===")
else:
    print(f"=== {tag}: no results parsed ===")
PY
