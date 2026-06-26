"""Day 3 - video data-curation pipeline: decode -> clip-sample -> tubelet -> loader.

Maps to the AMI JD bullet: "scalable infrastructure for video data processing
and curation." Profile clips/sec and find the bottleneck (decode vs IO vs transform).
"""
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from decord import VideoReader, cpu


class VideoClipDataset(Dataset):
    def __init__(self, root: str, clip_len: int = 16, stride: int = 4, size: int = 224, 
                 motion_thresh: float = 0.006, clips_per_video: int = 64):
        self.paths = sorted(glob.glob(f"{root}/**/*.mp4", recursive=True))
        self.clip_len, self.stride, self.size, self.motion_thresh = clip_len, stride, size, motion_thresh

        # CLIP SAMPLING DENSITY (a real curation decision): one long video holds thousands
        # of valid clip windows. We expose `clips_per_video` indices per file; _decode_clip's
        # random start makes each one a different window. 1-video-1-clip would be absurd.
        self.clips_per_video = clips_per_video

    def __len__(self):
        return len(self.paths)* self.clips_per_video

    def _decode_clip(self, path):
        # Goal: open the video, pick clip_len frame indices spaced by stride, decode just those.
        # num_threads = 1 is critical: the DataLoader already spawns num_workers
        # processes. Letting decord also multithread per-worker oversubscribes the
        # CPU and TANKS throughput. Parallelism lives at the worker level.

        vr = VideoReader(path, ctx = cpu(0), num_threads=1,
                         width = self.size, height = self.size) #resize at decode
        n = len(vr)
        span = self.clip_len * self.stride                                    #frames the clip spans in the source
        start = 0 if n <= span else np.random.randint(0, n - span)
        idx = (start + np.arange(self.clip_len) * self.stride).clip(0, n - 1) 
        frames = vr.get_batch(idx.tolist()).asnumpy()                         # (T, H, W, C) uint8
        return frames

    def _curate(self, frames) -> bool:
        # Goal: return False to drop near-static clips.
        # frames: (T, W, H, C) uint 8. Mean Absolute inter-frame difference  = a
        # cheap motion proxy. A locked-off / slideshow clip has ~0 motion -> trivial
        # temporal prediction -> drop it.

        f = frames.astype(np.float32) / 255.0  
        motion = np.abs(f[1:] - f[:-1]).mean() # avg pixel change between frames
        return motion > self.motion_thresh     # keep only if it actually moves

    def __getitem__(self, i):
        #Goal: (T,H,W,C) uint8 → normalized (C,T,H,W) float tensor.
        frames = self._decode_clip(self.paths[i % len(self.paths)])    # (T,H,W,C)
        # resize/normalize -> (C,T,H,W); tubelet-ify happens in the model/patch embed.
        
        if not self._curate(frames):
            #rejected: resample a different item rather than return a dead clip
            return self.__getitem__(np.random.randint(len(self)))
        
        x = torch.from_numpy(frames).float() / 255.0 # (T, H, W, C) in [0,1]
        x = x.permute(3, 0, 1, 2).contiguous()       # -> (C, T, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return (x-mean)/ std

def analyze_curation(root, n = 200, **kw):
    ds = VideoClipDataset(root, **kw)
    nv = len(ds.paths)
    motions = []

    for k in range(n):
        frames = ds._decode_clip(ds.paths[k % nv]) #round robin the files
        f = frames.astype(np.float32) / 255.0
        motions.append(np.abs(f[1:] - f[:-1]).mean())
    motions = np.array(motions)
    rej = (motions <= ds.motion_thresh).mean()
    print(f"rejection rate: {rej:.1%}  (thresh={ds.motion_thresh})")
    for p in (5, 25, 50, 75, 95):
        print(f"  p{p:>2} motion = {np.percentile(motions, p):.4f}")

def build_loader(root, batch_size=8, num_workers=8, **kw):
    ds = VideoClipDataset(root, **kw)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, drop_last=True)


def profile(loader, n_batches=50):
    import time
    t0 = time.time()
    seen = 0
    for i, batch in enumerate(loader):
        seen += batch.shape[0]
        if i + 1 >= n_batches:
            break
    dt = time.time() - t0
    print(f"{seen/dt:.1f} clips/sec over {dt:.1f}s  (bottleneck? compare num_workers)")


if __name__ == "__main__":
    samples = r"C:\Users\Koushik\vjepa2-probe\samples"
    profile(build_loader(samples))
    #analyze_curation(samples)
