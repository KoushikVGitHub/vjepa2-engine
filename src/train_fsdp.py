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
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from jepa_loss import JEPA, TinyEncoder, TinyPredictor, random_block_mask
from data.fields import FieldMapDataset


def setup():
    """
    Initializes the PyTorch distributed process group for multi-GPU communication.

    Behavior:
        Connects to the NCCL backend, reads the local rank from the environment 
        variables, and locks the current process to its corresponding GPU.
        
    Role in Program:
        Mandatory prerequisite for any DDP or FSDP operations.
    """
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    """
    Destroys the distributed process group.

    Role in Program:
        Cleanly shuts down inter-process communication at the end of training 
        to prevent zombie processes or hung terminal sessions.
    """
    dist.destroy_process_group()


def is_main():
    """
    Checks if the current process is the primary/master node (Rank 0).

    Role in Program:
        Used to filter operations that should only happen once globally, preventing 
        redundant logs or conflicting file writes across multiple GPUs.
    """
    return (not dist.is_initialized()) or dist.get_rank() == 0


def rprint(*a, **kw):
    """
    A rank-aware print function.

    Behavior:
        Wraps the standard Python `print` function but restricts execution so that 
        only the main process (Rank 0) actually outputs to standard out.
        
    Role in Program:
        Keeps console logs clean and readable during distributed training.
    """
    if is_main():
        print(*a, **kw, flush=True)


class JEPAStep(nn.Module):
    """
    Adapts JEPA to a standard ``forward(batch) -> loss`` interface.

    Behavior:
        Samples a fresh random block mask during every forward pass. It manages the 
        routing of the distributed SIGReg generator args and exposes a shard-aligned 
        EMA update step for the target encoder.
        
    Role in Program:
        Abstracts the complex multi-stage JEPA logic (context, targets, masking) 
        into a clean, standard PyTorch Module API that FSDP and simple training 
        loops can easily digest.
    """

    def __init__(self, jepa: JEPA, grid: int, block: int):
        super().__init__()
        self.jepa = jepa
        self.grid = grid
        self.block = block

    def forward(self, x, sigreg_generator=None, sigreg_distributed=False):
        ctx_idx, tgt_idx = random_block_mask(self.grid, self.block, x.device)
        # sigreg_* thread down to _forward_lejepa (ignored in ema mode).
        loss, _ = self.jepa(x, ctx_idx, tgt_idx,
                            sigreg_generator=sigreg_generator,
                            sigreg_distributed=sigreg_distributed)
        return loss

    @torch.no_grad()
    def step_ema(self):
        if self.jepa.loss_mode != "ema":          # lejepa has no teacher -> no-op
            return
        d = self.jepa.ema_decay
        for pt, po in zip(self.jepa.target_encoder.parameters(),
                          self.jepa.context_encoder.parameters()):
            pt.mul_(d).add_(po, alpha=1 - d)


def build_model(args, device):
    """
    Instantiates the raw, unsharded neural network components.

    Behavior:
        Constructs the TinyEncoder, TinyPredictor, and JEPA wrapper based on the 
        hyperparameters specified in `args`. Wraps them in `JEPAStep` and moves 
        them to the specified device.
        
    Role in Program:
        Provides the baseline architecture before distributed data parallel or 
        sharding strategies are applied.
    """

    enc = TinyEncoder(img=args.img, patch=args.patch, d=args.d,
                      heads=args.heads, layers=args.layers)
    grid = args.img // args.patch
    pred = TinyPredictor(grid * grid, d=args.d, heads=args.heads, layers=args.layers)
    jepa = JEPA(enc, pred, ema_decay=0.998, stop_grad=True,
                loss_mode=args.loss, sigreg_lambda=args.sigreg_lambda)
    return JEPAStep(jepa, grid, block=args.block).to(device)


def wrap_fsdp(model, args, device):
    """
    Wraps the baseline model in Fully Sharded Data Parallel (FSDP).

    Behavior:
        Configures an auto-wrap policy to shard at the Transformer block boundaries. 
        Enables mixed-precision (bf16) if requested to save memory and boost speed. 
        Optionally applies activation checkpointing to discard and recompute activations, 
        further reducing VRAM pressure at the cost of slight compute overhead.
        
    Role in Program:
        Scales the architecture by distributing model parameters, gradients, and 
        optimizer states across multiple GPUs, allowing the training of models that 
        exceed single-GPU memory limits.
    """
    
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
    """
    Wraps the baseline model in standard Distributed Data Parallel (DDP).

    Behavior:
        Replicates the entire model across all GPUs and synchronizes gradients during 
        the backward pass. Can also apply activation checkpointing if requested.
        
    Role in Program:
        Acts as the standard scaling baseline. Unlike FSDP, it does not shard parameters 
        across GPUs, meaning it requires higher VRAM but avoids parameter communication 
        overhead during the forward/backward passes.
    """

    if args.ckpt:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, nn.TransformerEncoderLayer),
        )
    return DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])


def save_checkpoint(model, args, path):
    """Consolidate params to rank 0 and save a full (unsharded) state dict.

    FSDP shards params across ranks, so we must gather a FULL_STATE_DICT (offloaded to CPU,
    rank0-only) before saving -- a plain model.state_dict() under FSDP returns only this rank's
    shard. Keys come out as the original hierarchy (jepa.context_encoder.*), which the probe's
    load_frozen_encoder rebuilds by splitting on 'context_encoder.'.
    """
    if args.mode == "fsdp":
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state = model.state_dict()
    else:  # ddp: unwrap the replica
        state = model.module.state_dict()
    if is_main():
        torch.save({"model": state, "args": vars(args)}, path)
        rprint(f"[ckpt] saved -> {path}")


def online_param_count(model):
    """
    Calculates the total number of trainable parameters in the active model.

    Role in Program:
        Provides the critical scale metric `n_online` needed by `estimate_mfu` 
        to calculate theoretical FLOP throughput. Ignores frozen target encoders.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_mfu(args, n_online, step_time):
    """
    Approximates Model-FLOPs Utilization (MFU) based on theoretical hardware maximums.

    Behavior:
        Uses the 6*N*tokens transformer rule of thumb. Adjusts the calculation 
        depending on `loss_mode` (EMA requires a frozen target forward pass, LeJEPA 
        requires two online forward passes). Divides the achieved FLOPs by the hardware's 
        peak theoretical TFLOPs.
        
    Role in Program:
        Provides a standardized hardware efficiency metric, quantifying how well the 
        model is saturating the GPU compute capabilities (typically 40-60% is excellent).
    """

    grid = args.img // args.patch
    tokens = args.batch * grid * grid
    if args.loss == "lejepa":
        flops_step = (6 * n_online + 2 * n_online) * tokens
    else:
        n_target = n_online // 2 if n_online else 0
        flops_step = (6 * n_online + 2 * n_target) * tokens
    achieved = flops_step / step_time
    return achieved / (args.peak_tflops * 1e12)


def infinite(loader, sampler):
    """
    Converts a finite PyTorch DataLoader into an infinite generator.

    Behavior:
        Yields batches endlessly. When a dataset is exhausted, it increments the 
        epoch counter on the `DistributedSampler` to ensure every rank's new shuffle 
        is decorrelated yet deterministically reproducible, and restarts iteration.
        
    Role in Program:
        Simplifies step-based (rather than epoch-based) training loops by abstracting 
        away dataset boundaries.
    """
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def build_dataloader(args, world, rank):
    """
    Configures the real CAMELS multifield dataloader, sharded across distributed ranks.

    Behavior:
        Loads physical simulation fields, applies log/asinh transformations based on 
        the field type, and pools them into a ConcatDataset. It uses a DistributedSampler 
        to ensure each rank sees a disjoint 1/world slice of the data.
        
    Role in Program:
        Feeds the true scientific data into the pipeline. Uses `drop_last=True` strictly 
        so that per-rank batches remain perfectly EQUAL, allowing the SIGReg loss to 
        safely rely on an exact global count of N = batch * world.
    """
    # analyze_all.py (all 13 fields) showed EVERY field is positive-definite -> uniform log10
    # (Vgas/Vcdm raw-min = +5 => velocity MAGNITUDE, not signed; MgFe +0.013). No field has a
    # degenerate/near-constant population (even p5 std is healthy), so min_std=0.05 is a defensive
    # floor that only catches a pathological fully-empty map (Mstar/Z/ne have raw-min 0). B is
    # dropped (IllustrisTNG-only + floor-dominated, p50 std 0.009). The 12 kept fields exist in
    # BOTH suites -> reusable for the SIMBA held-out probe.
    FIELDS = ["Mgas", "Mcdm", "Mtot", "Mstar", "T", "P", "Z", "HI", "ne", "MgFe", "Vgas", "Vcdm"]
    field_configs = [
        {"npy_path": os.path.join(args.data_root, f"Maps_{f}_IllustrisTNG_LH_z=0.00.npy"),
         "name": f, "transform": "log10", "min_std": 0.05}
        for f in FIELDS
    ]

    # Keep only fields whose file is actually on the volume, so a partial download still runs
    # (and newly-transferred fields auto-join on the next launch).
    missing = [c["name"] for c in field_configs if not os.path.exists(c["npy_path"])]
    field_configs = [c for c in field_configs if os.path.exists(c["npy_path"])]
    if missing:
        rprint(f"[data] skipping fields with no file under {args.data_root}: {missing}")
    if not field_configs:
        raise FileNotFoundError(f"no CAMELS field files found under {args.data_root}")
    rprint(f"[data] pooling fields: {[c['name'] for c in field_configs]}")

    # 2. Pool all specified field files into ONE large corpus
    ds = ConcatDataset([
        FieldMapDataset(size=args.img, **config) for config in field_configs
    ])

    # 3. Create the distributed sampler
    # drop_last=True is CRITICAL here to ensure batch sizes are identical 
    # across all ranks for the SIGReg global expectation step.
    sampler = DistributedSampler(
        ds, 
        num_replicas=world, 
        rank=rank,
        shuffle=True, 
        drop_last=True
    )

    # 4. Construct the DataLoader
    # drop_last=True is also required on the DataLoader level to prevent
    # the final step of the epoch from having a smaller local batch size.
    loader = DataLoader(
        ds, 
        batch_size=args.batch, 
        sampler=sampler,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True
    )

    return loader, sampler


def main():
    args = parse_args()
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed)
    world = dist.get_world_size()

    model = build_model(args, device)
    n_online = online_param_count(model)
    model = wrap_fsdp(model, args, device) if args.mode == "fsdp" else wrap_ddp(model, args)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    use_autocast = args.bf16 and args.mode == "ddp"

    # Real CAMELS multifield data, sharded across ranks (replaces synthetic make_batch).
    loader, sampler = build_dataloader(args, world, dist.get_rank())
    data = infinite(loader, sampler)

    # One generator object per rank; RESEEDED identically each step so V/t match across ranks.
    sigreg_gen = torch.Generator(device=device)
    lejepa = (args.loss == "lejepa")

    model.train()
    warmup = min(10, args.steps // 2)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = None
    for step in range(args.steps):
        if step == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        batch = next(data).to(device, non_blocking=True)          # (B, 1, H, W)

        # Ensure per-step SYNCED seed so every rank draws the SAME projections V and
        # frequencies t -- else the all-reduced sums are incomparable.
        if lejepa: 
            sigreg_gen.manual_seed(args.seed + step)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            loss = model(batch,
                         sigreg_generator=sigreg_gen if lejepa else None,
                         sigreg_distributed=lejepa)
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

    if args.save:
        save_checkpoint(model, args, args.save)   # FSDP gathers to rank0 (barrier); all ranks call
    cleanup()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ddp", "fsdp"], default="fsdp")
    p.add_argument("--loss", choices=["ema", "lejepa"], default="lejepa")
    p.add_argument("--sigreg-lambda", type=float, default=0.02)
    p.add_argument("--data-root", type=str, default="/workspace/data",
                   help="dir holding Maps_<field>_<suite>_LH_z=0.00.npy")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--save", type=str, default="",
                   help="path to save a gathered FULL_STATE_DICT checkpoint (empty = skip)")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--ckpt", action="store_true", help="activation checkpointing")
    p.add_argument("--img", type=int, default=256, help="CAMELS 2D maps are 256x256")
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
