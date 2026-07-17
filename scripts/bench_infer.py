"""Day 5 / Stage 4 — inference-optimization benchmark for the OWN ViT-L encoder.

Measures forward-only inference on `jepa_loss.ViTEncoder` (keeper config
img256/patch16/d1024/heads16/L24, 256 tokens) lever-by-lever, single GPU, and prints a
markdown results table you paste into study/notes/day5_inference.md.

Levers (see the P1-P5 predictions in the notes BEFORE you run this):
  baseline     eager, fp32                         -- the reference
  bf16         autocast bf16                        -- precision lever  (P2)
  bf16+flash   bf16 + forced FlashAttention SDPA     -- fused attention (P3)
  bf16+compile bf16 + torch.compile                  -- graph capture   (P4)
  int8         dynamic int8 PTQ on nn.Linear         -- quant lever     (P5b)

Throughput/latency levers report p50/p99 latency, images/sec, peak mem, MFU. Accuracy-affecting
levers (bf16, int8) additionally report cosine drift of the pooled embedding vs the fp32 reference.

Usage:
  python scripts/bench_infer.py --batch 64 --iters 50 --warmup 10
  python scripts/bench_infer.py --lever baseline --batch-sweep 1,8,64,256   # P5a batching sweep
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from jepa_loss import ViTEncoder

# ViT-L keeper config -- keep in sync with run_probe.ENC so we benchmark the real encoder shape.
ENC = dict(img=256, patch=16, d=1024, heads=16, layers=24)

LEVERS = ["baseline", "bf16", "bf16+flash", "bf16+compile", "int8"]


def build_encoder(device):
    """
    Instantiates the ViT encoder with randomized weights for hardware benchmarking.
    
    Behavior:
        Initializes the model in eval mode and freezes the parameters to mimic 
        a strict deployment/inference environment.
        
    Role in Program:
        Provides the raw workload to be optimized. We don't need trained weights 
        because we are measuring raw hardware compute limits and mathematical 
        drift, not semantic accuracy.
    """
    enc = ViTEncoder(**ENC).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def pool(feats):  # (B, n, d) -> (B, d), same pooling the probe uses
    """
    Collapses the token sequence into a single global representation.
    
    Behavior:
        Averages the sequence dimension: (B, n, d) -> (B, d).
        
    Role in Program:
        Matches the exact pooling operation the downstream probe uses. Vital for 
        measuring the true cosine drift caused by lower-precision arithmetic.
    """
    return feats.mean(dim=1)


def flops_per_image(enc_config):
    """
    Analytically calculates the theoretical FLOPs for one forward pass of the ViT.
    
    Behavior:
        Computes matmul FLOPs via the `2 * N_params * tokens` rule of thumb. 
        Calculates N_params for a ViT layer (Attention = 4d^2, MLP = 8d^2 -> 12d^2). 
        Adds the exact FLOP count for calculating the raw self-attention scores 
        (2 * Layers * tokens^2 * d).
        
    Role in Program:
        Provides the mathematical denominator required to calculate Model FLOPs 
        Utilization (MFU).
    """
    tokens = (enc_config['img'] // enc_config['patch'])**2
    d = enc_config['d']
    layers = enc_config['layers']

    # Trainable matmul params per layer: attention Q/K/V/O = 4*d^2, MLP = 4*d^2 (this encoder uses
    # dim_feedforward = 2*d, so Linear(d,2d)+Linear(2d,d) = 4*d^2 -- NOT the textbook 8*d^2 that a
    # 4*d MLP would give). Total 8*d^2 per layer, not the standard-config 12*d^2.
    n_params = layers * 8 * (d ** 2)

    # Base FLOPs (Matrix Multiplications)
    matmul_flops = 2 * n_params * tokens

    # Attention Score FLOPs (Q * K^T and Attention * V)
    attn_flops = 2 * layers * (tokens ** 2) * d
    return matmul_flops + attn_flops


def make_runner(lever, enc, device):
    """
    Wraps the baseline encoder in specific hardware-acceleration contexts.
    
    Behavior:
        Dispatches to one of five optimization "levers":
        1. baseline: FP32 eager.
        2. bf16: Applies PyTorch AMP autocast.
        3. bf16+flash: Forces the SDP backend to use highly fused FlashAttention kernels.
        4. bf16+compile: Compiles the computational graph via TorchDynamo.
        5. int8: Applies dynamic Post-Training Quantization (PTQ) for CPU deployment.
        
    Role in Program:
        The core of the benchmark. This function isolates the exact mechanism 
        being tested so the timing harness can treat them all agnostically.
    """
    if lever == "baseline":
        def runner(x):
            return pool(enc(x))
        return runner, False

    if lever == "bf16":
        def runner(x):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return pool(enc(x))
        return runner, False

    if lever == "bf16+flash":
        def runner(x):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return pool(enc(x))
        return runner, False

    if lever == "bf16+compile":
        # Compile ONCE outside the runner. max-autotune extracts peak performance 
        # but takes significantly longer during the warmup phase.
        cenc = torch.compile(enc, mode="max-autotune")
        def runner(x):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return pool(cenc(x))
        return runner, True

    if lever == "int8":
        # Dynamic int8 quantization acts on nn.Linear weights directly.
        # It executes primarily on the CPU.
        qenc = torch.ao.quantization.quantize_dynamic(enc, {nn.Linear}, dtype=torch.qint8)
        def runner(x):
            # No autocast here - dynamic int8 handles its own internal fp32/int8 casting.
            return pool(qenc(x))
        return runner, False

    raise ValueError(f"unknown lever {lever!r}")


# ---------------------------------------------------------------------------
# Timing harness -- identical for every lever. CUDA events measure device time (not wall clock,
# which includes Python dispatch stalls the events skip past on the GPU stream).
# ---------------------------------------------------------------------------

@torch.no_grad()
def time_runner(runner, x, iters, warmup, device):
    """
    Micro-benchmarking harness using accurate device synchronization.
    
    Behavior:
        Executes a warmup phase to absorb CUDA initialization, cuDNN autotuning, 
        and JIT compilation overheads. Then, loops `iters` times wrapping the 
        runner in precise `torch.cuda.Event` markers (or `time.perf_counter` for CPU).
        Returns a dictionary of strict percentile latencies, throughput, and memory.
        
    Role in Program:
        Provides the statistically rigorous measurement framework ensuring the 
        reported optimization gains are actual GPU compute wins, not Python overhead.
    """

    is_cuda = device.startswith("cuda")

    for _ in range(warmup):                    # warmup: cuDNN autotune, compile, allocator warm
        runner(x)
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    lat_ms = []
    for _ in range(iters):
        if is_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            runner(x)
            end.record()
            end.synchronize()
            lat_ms.append(start.elapsed_time(end))     # per-forward device ms
        else:
            import time
            t0 = time.perf_counter()
            runner(x)
            lat_ms.append((time.perf_counter() - t0) * 1e3)

    lat_ms.sort()
    p50 = statistics.median(lat_ms)
    p99 = lat_ms[min(len(lat_ms) - 1, int(round(0.99 * len(lat_ms))) - 1)]
    batch = x.size(0)
    imgs_per_sec = batch / (statistics.mean(lat_ms) / 1e3)
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if is_cuda else float("nan")
    return dict(p50=p50, p99=p99, imgs_per_sec=imgs_per_sec, peak_gb=peak_gb)


def mfu(imgs_per_sec, peak_tflops):
    """
    Calculates Model FLOPs Utilization.
    
    Behavior:
        Computes the achieved FLOPs per second (Throughput * FLOPs per image) and 
        divides it by the GPU's hardware theoretical peak TFLOPs.
        
    Role in Program:
        Yields the ultimate efficiency metric (e.g., "This lever uses 54% of the GPU").
    """
    return (imgs_per_sec * flops_per_image(ENC)) / (peak_tflops * 1e12) * 100.0


def main():
    """
    Orchestrates the entire inference benchmark matrix.
    
    Behavior:
        Generates dummy reference data, establishes an FP32 exact mathematical 
        baseline, and iterates over the requested `levers` and `batches`. It 
        calculates throughput limits and measures the mathematical accuracy penalty 
        (cosine drift) induced by reducing precision. Dumps a clean Markdown table.
        
    Role in Program:
        The entry point that ties the configuration parsing, the optimization 
        wrappers, and the timing harness together to produce the final deliverable.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", choices=LEVERS + ["all"], default="all")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--batch-sweep", default=None, help="e.g. 1,8,64,256 -- runs --lever at each batch (P5a)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--peak-tflops", type=float, default=150.0, help="A40 bf16 dense peak")
    args = ap.parse_args()

    levers = LEVERS if args.lever == "all" else [args.lever]
    batches = [int(b) for b in args.batch_sweep.split(",")] if args.batch_sweep else [args.batch]

    # fp32-eager reference pooled embedding for cosine-drift accounting (fixed weights + input).
    enc_ref = build_encoder(args.device)
    x_ref = torch.randn(args.batch, 1, ENC["img"], ENC["img"], device=args.device)
    with torch.no_grad():
        ref_emb = pool(enc_ref(x_ref)).float()

    rows = []
    for lever in levers:
        # int8 dynamic quant runs on CPU -- honor that (see make_runner SPEC).
        dev = "cpu" if lever == "int8" else args.device
        if lever == "int8":
            # SAME weights as the reference (moved to CPU) so cosine drift isolates quantization
            # error -- not the gap between two different random inits.
            enc = build_encoder("cpu")
            enc.load_state_dict({k: v.cpu() for k, v in enc_ref.state_dict().items()})
        else:
            enc = enc_ref  # reuse the reference weights -> drift = pure precision effect
        runner, _ = make_runner(lever, enc, dev)
        for b in batches:
            x = torch.randn(b, 1, ENC["img"], ENC["img"], device=dev)
            stats = time_runner(runner, x, args.iters, args.warmup, dev)

            drift = ""
            if lever in ("bf16", "bf16+flash", "int8") and b == args.batch:
                with torch.no_grad():
                    emb = runner(x_ref.to(dev)).float().to(ref_emb.device)
                cos = F.cosine_similarity(emb, ref_emb).mean().item()
                drift = f"{1 - cos:.2e}"

            try:
                mfu_val = f"{mfu(stats['imgs_per_sec'], args.peak_tflops):.1f}"
            except NotImplementedError:
                mfu_val = "TODO"
            rows.append((f"{lever}@b{b}", stats, mfu_val, drift))

    # markdown table -> paste into day5_inference.md
    print("\n| lever | latency p50 (ms) | latency p99 (ms) | images/sec | peak mem (GB) | MFU (%) | cosine drift |")
    print("|-------|------------------|------------------|-----------|---------------|---------|--------------|")
    for label, s, mfu_val, drift in rows:
        print(f"| {label} | {s['p50']:.2f} | {s['p99']:.2f} | {s['imgs_per_sec']:.1f} | "
              f"{s['peak_gb']:.2f} | {mfu_val} | {drift or '-'} |")


if __name__ == "__main__":
    main()
