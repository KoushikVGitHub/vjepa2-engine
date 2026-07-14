# Day 4 — Distributed training sweep

**Model:** JEPA with ViT-B-scale encoders (`--d 768 --layers 12 --heads 12 --img 224 --patch 16`),
114.5M trainable params (context encoder + predictor; frozen EMA target excluded).
**Hardware:** 2× NVIDIA A40 (bf16 dense peak 150 TFLOPS), single node, NCCL over SHM.
**Command:** `torchrun --standalone --nproc_per_node=2 src/train_fsdp.py --mode … --steps 100 --peak-tflops 150`

| # | mode | bf16 | ckpt | samples/sec (global) | sec/step (ms) | peak mem / GPU (GB) | MFU (%) |
|---|------|------|------|----------------------|---------------|----------------------|---------|
| 1 | ddp  | no   | no   | 146.3                | 874.9         | 12.11                | 7.66    |
| 2 | fsdp | no   | no   | 145.5                | 879.8         | 10.88                | 7.62    |
| 3 | fsdp | yes  | no   | 491.5                | 260.4         | 5.46                 | 25.74   |
| 4 | fsdp | yes  | yes  | 408.5                | 313.4         | 1.62                 | 21.40   |

## Observations
- **FSDP vs DDP, fp32 (1 → 2):** throughput is identical (146 vs 146 samples/sec) while peak
  memory drops ~10% (12.11 → 10.88 GB). At this model size FSDP's gather/reshard overhead is
  negligible and the sharding win is modest, because *activations*, not parameters, dominate
  memory here. FSDP earns its keep at scales where the model barely fits — not at 114M.
- **bf16 (2 → 3):** the dominant lever. **3.4× throughput** (145 → 492 samples/sec), step time
  880 → 260 ms, MFU 7.6% → 25.7%, and memory halved (10.88 → 5.46 GB). The fp32 rows were
  compute-bound on the A40 (fp32 runs on CUDA cores ~37 TFLOPS); bf16 engages the tensor cores
  (150 TFLOPS), which is why the speed-up exceeds the textbook 2×.
- **Activation checkpointing (3 → 4):** the memory lever. Peak memory drops **3.4×**
  (5.46 → 1.62 GB) for a ~20% throughput cost (492 → 409 samples/sec) — the classic
  recompute-for-memory trade. This is what you pull when you need to fit a larger model or batch.
- **End to end (1 → 4):** DDP/fp32 baseline → FSDP+bf16+ckpt = **2.8× throughput and 7.5× less
  peak memory** (12.11 → 1.62 GB).
- **MFU ceiling ~26%:** even the best config sits below the healthy 40–55% band. Likely causes:
  `nn.TransformerEncoderLayer` not using a fused/FlashAttention kernel, single-thread CPU dispatch
  (`OMP_NUM_THREADS=1`), and a modest batch (64). This points directly at the Day-5 lever
  (`torch.compile` + FlashAttention) — i.e. the harness is comms/memory-healthy; the remaining
  gap is kernel efficiency, not distribution.

## Notes
- MFU is an analytical estimate (`6·N·tokens` for the online path, `2·N` fwd-only for the
  frozen EMA target), measured against the A40's 150 TFLOPS bf16 peak.
- NCCL required `NCCL_P2P_DISABLE=1` + `NCCL_HOSTID=samehost` (and clearing the image's
  baked-in `NCCL_SHM_DISABLE`) to use the fast intra-node shared-memory transport instead of
  falling back to TCP sockets.
