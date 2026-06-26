# Day 1 — Environment setup (pre-staged)

**Day-1 goal:** clone V-JEPA 2, run inference on 2–3 clips, extract embeddings, compute
pairwise cosine similarity. The learning point (P4 prediction): *similar clips should have
higher cosine similarity than dissimilar clips* — confirm or be surprised.

## Compute decision (make this first)
Inference + probing fits a **single GPU** — Colab Pro or Kaggle is enough for Day 1.
(Save RunPod / Lambda 2×GPU rental for Day 4's FSDP run.)

| Option | Notes |
|---|---|
| **Colab Pro** | Easiest; T4/L4/A100 depending on tier. Mount Drive, clone repo, run setup. |
| **Kaggle** | Free P100/T4 with weekly GPU quota; persistent /kaggle/working. |
| **Local** | Only if you have an NVIDIA GPU; this machine is Windows — use WSL2 for the bash script. |

## Run the setup script (on the GPU box)
```bash
cd vjepa2-probe
bash scripts/setup_day1.sh
```
It will: make a venv, install `requirements.txt`, clone `facebookresearch/vjepa2` into
`external/`, record the commit hash to `notes/vjepa2_commit.txt`, and run a GPU sanity check.

### On Colab (no venv needed)
```python
!git clone <your repo url> vjepa2-probe
%cd vjepa2-probe
!pip install -r requirements.txt
!git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
!pip install -e external/vjepa2
import torch; print(torch.cuda.get_device_name(0))
```

## Sample clips
Put 2–3 short `.mp4` files in `samples/`. For the cosine-sim check pick **2 visually similar
clips + 1 clearly different** so the similarity ordering is meaningful.

## Then: implement the two TODOs in `src/infer.py`
- `load_model(device)` — load the pretrained V-JEPA 2 encoder (entrypoint name per upstream repo), `.eval()`.
- `embed_video(model, path, device)` — decode → preprocess → encoder → pooled 1D embedding.
Run: `python src/infer.py --video samples/a.mp4 --video2 samples/b.mp4`

## Day-1 PRE-TEST (do this CLOSED-BOOK before studying — don't peek)
- P1. Why is predicting future *pixels* a bad objective? What does predicting in representation space buy you?
- P2. What is "representation collapse" and why would a naive JEPA suffer it?
- P3. How does JEPA differ from (a) a VAE, (b) contrastive learning like SimCLR?
- P4. *Predict:* cosine similarity for 2 similar vs 2 different clips — what will you see?

## Spaced review to fold into Day 1 (all rated 3/5 on Day 0)
- Two-paradigm split (render-to-predict vs compress-to-understand)
- AMI's compression case
- The strategic argument against the bet
