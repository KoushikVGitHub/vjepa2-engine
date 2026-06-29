"""Distributed training harness for the JEPA model (PyTorch FSDP / DDP, bf16).

Measures throughput, peak memory, and approximate MFU across configurations to quantify
the effect of FSDP, bf16 mixed precision, and activation checkpointing.

    torchrun --standalone --nproc_per_node=2 src/train_fsdp.py \
        --mode fsdp --bf16 --ckpt --steps 200 --peak-tflops 150
"""
import os
import time
import argparse
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

from jepa_loss import JEPA, TinyEncoder, TinyPredictor, random_block_mask


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def rprint(*a, **kw):
    if is_main():
        print(*a, **kw, flush=True)


class JEPAStep(nn.Module):
    """Adapts JEPA to a ``forward(batch) -> loss`` interface, sampling a fresh block mask
    each step and exposing a shard-aligned EMA update for the target encoder."""

    def __init__(self, jepa: JEPA, grid: int, block: int):
        super().__init__()
        self.jepa = jepa
        self.grid = grid
        self.block = block

    def forward(self, x):
        ctx_idx, tgt_idx = random_block_mask(self.grid, self.block, x.device)
        loss, _ = self.jepa(x, ctx_idx, tgt_idx)
        return loss

    @torch.no_grad()
    def step_ema(self):
        d = self.jepa.ema_decay
        for pt, po in zip(self.jepa.target_encoder.parameters(),
                          self.jepa.context_encoder.parameters()):
            pt.mul_(d).add_(po, alpha=1 - d)


def build_model(args, device):
    enc = TinyEncoder(img=args.img, patch=args.patch, d=args.d,
                      heads=args.heads, layers=args.layers)
    grid = args.img // args.patch
    pred = TinyPredictor(grid * grid, d=args.d, heads=args.heads, layers=args.layers)
    jepa = JEPA(enc, pred, ema_decay=0.998, stop_grad=True)
    return JEPAStep(jepa, grid, block=args.block).to(device)


def wrap_fsdp(model, args, device):
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={nn.TransformerEncoderLayer, TinyEncoder, TinyPredictor},
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    ) if args.bf16 else None

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        sync_module_states=True,
        device_id=device,
    )

    if args.ckpt:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, nn.TransformerEncoderLayer),
        )
    return model


def wrap_ddp(model, args):
    if args.ckpt:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, nn.TransformerEncoderLayer),
        )
    return DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])


def online_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_mfu(args, n_online, step_time):
    """Approximate model-FLOPs utilization via the 6*N*tokens transformer rule (fwd+bwd for
    the online path, fwd only for the frozen EMA target)."""
    grid = args.img // args.patch
    tokens = args.batch * grid * grid
    n_target = n_online // 2 if n_online else 0
    flops_step = (6 * n_online + 2 * n_target) * tokens
    achieved = flops_step / step_time
    return achieved / (args.peak_tflops * 1e12)


def main():
    args = parse_args()
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234)
    world = dist.get_world_size()

    model = build_model(args, device)
    n_online = online_param_count(model)
    model = wrap_fsdp(model, args, device) if args.mode == "fsdp" else wrap_ddp(model, args)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    use_autocast = args.bf16 and args.mode == "ddp"

    def make_batch():
        return torch.randn(args.batch, 1, args.img, args.img, device=device)

    model.train()
    warmup = min(10, args.steps // 2)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = None
    for step in range(args.steps):
        if step == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        batch = make_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            loss = model(batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        (model.module if hasattr(model, "module") else model).step_ema()
        if step % 20 == 0:
            rprint(f"step {step:>4} loss {loss.item():.4f}")

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    n_steps = args.steps - warmup
    sps = (n_steps * args.batch * world) / elapsed
    step_time = elapsed / n_steps
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    mfu = estimate_mfu(args, n_online, step_time)

    rprint("\n=== RESULT ===========================================")
    rprint(f" mode={args.mode} bf16={args.bf16} ckpt={args.ckpt} "
           f"world={world} online_params={n_online/1e6:.1f}M")
    rprint(f" samples/sec (global) : {sps:8.1f}")
    rprint(f" sec/step (per rank)  : {step_time*1e3:8.1f} ms")
    rprint(f" peak mem / gpu       : {peak_mem:8.2f} GB")
    rprint(f" MFU (approx)         : {mfu*100:8.2f} %")
    rprint("======================================================")
    cleanup()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ddp", "fsdp"], default="fsdp")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--ckpt", action="store_true", help="activation checkpointing")
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--block", type=int, default=4, help="target block size on the patch grid")
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--batch", type=int, default=64, help="per-GPU batch")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--peak-tflops", type=float, default=312.0,
                   help="GPU bf16 dense peak: A100=312, H100=990, A40=150, 3090=71")
    return p.parse_args()


if __name__ == "__main__":
    main()
