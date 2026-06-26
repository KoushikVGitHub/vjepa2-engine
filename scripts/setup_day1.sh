#!/usr/bin/env bash
# Day-1 environment setup for vjepa2-probe.
# Run on a GPU box (Colab Pro / Kaggle / RunPod). Idempotent-ish: safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> 1. Python venv"
python -m venv .venv || true
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> 2. Project deps"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> 3. Clone V-JEPA 2 (official)"
if [ ! -d external/vjepa2 ]; then
  mkdir -p external
  git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
fi
# Pin the commit you actually used so the repro is reproducible:
( cd external/vjepa2 && git rev-parse HEAD | tee ../../notes/vjepa2_commit.txt )
# Install it (follow upstream README if this errors; some versions use a different entrypoint):
pip install -e external/vjepa2 || echo "WARN: editable install failed — check upstream install instructions."

echo "==> 4. Sample clips"
mkdir -p samples
echo "Place 2-3 short .mp4 clips in $REPO_ROOT/samples/ (pick 2 visually similar + 1 different for the cosine-sim check)."

echo "==> 5. GPU sanity check"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "==> Done. Next: implement load_model/embed_video in src/infer.py, then run the cosine-sim check."
