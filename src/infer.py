"""Day 1 — load pretrained V-JEPA 2, run inference, extract & compare embeddings.

Goal: prove you can load the model and that embeddings behave sensibly
(similar clips -> HIGHER cosine similarity than dissimilar clips).

Uses HuggingFace transformers (no repo clone needed for inference):
    model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    feats = model.get_vision_features(**processor(frames, return_tensors="pt"))

Run:
    python src/infer.py --video samples/a.mp4 --video2 samples/b.mp4
"""
import argparse
import av
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor

# fpc64 = the checkpoint expects 64 frames per clip. Keep this in sync with the repo id.
HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 64


def load_model(device: str):
    """Load the pretrained V-JEPA 2 encoder + its video processor, in eval mode."""
    model = AutoModel.from_pretrained(HF_REPO).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(HF_REPO)
    return model, processor


def _sample_frames(path: str, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    """Decode a video (PyAV) and evenly sample `num_frames` -> uint8 tensor (T, C, H, W).

    PyAV avoids torchvision.io/torchaudio, which mismatch easily on hosted GPU runtimes.
    """
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]  # list of (H,W,C)
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    video = np.stack(frames)                                          # (T,H,W,C) uint8
    idx = np.linspace(0, video.shape[0] - 1, num_frames).round().astype(int)  # even sampling
    return torch.from_numpy(video[idx]).permute(0, 3, 1, 2).contiguous()      # (num_frames,C,H,W)


@torch.no_grad()
def embed_video(model, processor, path: str, device: str) -> torch.Tensor:
    """Return a 1D pooled embedding for the clip (mean over patch/token dim)."""
    frames = _sample_frames(path)
    inputs = processor(frames, return_tensors="pt").to(device)
    feats = model.get_vision_features(**inputs)  # (B, num_tokens, hidden)
    pooled = feats.mean(dim=1).squeeze(0)        # (hidden,)
    return pooled.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video2", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, processor = load_model(args.device)
    e1 = embed_video(model, processor, args.video, args.device)
    print("embedding shape:", tuple(e1.shape))

    if args.video2:
        e2 = embed_video(model, processor, args.video2, args.device)
        sim = F.cosine_similarity(e1[None], e2[None]).item()
        print(f"cosine similarity: {sim:.4f}")  # check against your Day-1 P4 prediction


if __name__ == "__main__":
    main()
