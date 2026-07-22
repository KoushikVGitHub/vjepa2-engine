"""Distributed preflight: prove torchrun actually spun up one rank PER GPU and that the ranks can
talk to each other (NCCL all-reduce), BEFORE burning A40 hours on a training run that silently
falls back to a single GPU.

Two failure modes this catches:
  1. torchrun launched fewer ranks than GPUs (wrong --nproc_per_node) -> world_size != #GPUs.
  2. Ranks launched but each pinned to the SAME device, or NCCL comms broken (the container
     SHM/P2P quirks that bit the Day-4 run) -> the all-reduce hangs or gives the wrong sum.

Run (on a 2-GPU pod):
    torchrun --standalone --nproc_per_node=2 scripts/check_dist.py

Expect: two lines (rank 0 + rank 1), each on a DISTINCT cuda:N with its own GPU name, and a final
"[rank 0] PASS" showing the all-reduced tensor == world_size (every rank contributed 1.0).
Pass --expect N to hard-assert a specific world size (e.g. --expect 2).
"""
import argparse
import datetime
import os

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", type=int, default=0,
                    help="hard-fail unless world_size == this (0 = just report)")
    args = ap.parse_args()

    # 10-min timeout so a broken NCCL setup ERRORS instead of hanging the pod forever.
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    name = torch.cuda.get_device_name(local_rank)
    total_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    # Ordered print so the two rank lines don't interleave mid-line.
    for r in range(world):
        if r == rank:
            print(f"[rank {rank}/{world}] local_rank={local_rank} device={device} "
                  f"gpu='{name}' {total_gb:.0f}GB visible_devices={visible} "
                  f"host={os.uname().nodename if hasattr(os, 'uname') else 'n/a'}",
                  flush=True)
        dist.barrier()

    # The real proof of PARALLEL execution: each rank contributes 1.0; a correct cross-GPU
    # all-reduce(SUM) yields exactly world_size. Wrong value or a hang == comms are broken.
    x = torch.ones(1, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    ok = abs(x.item() - world) < 1e-6

    if rank == 0:
        gpus_seen = torch.cuda.device_count()
        print(f"\n[rank 0] world_size={world} | GPUs visible to this process={gpus_seen} | "
              f"all_reduce(ones)={x.item():.1f} (expected {world})", flush=True)
        if args.expect and world != args.expect:
            print(f"[rank 0] FAIL: expected world_size={args.expect}, got {world}. "
                  f"Check --nproc_per_node and CUDA_VISIBLE_DEVICES.", flush=True)
        elif not ok:
            print("[rank 0] FAIL: all-reduce did not sum across ranks -- NCCL comms broken.",
                  flush=True)
        elif world < 2:
            print("[rank 0] WARN: world_size=1 -- running on a SINGLE GPU (not parallel). "
                  "Relaunch with --nproc_per_node=2 on a 2-GPU pod.", flush=True)
        else:
            print(f"[rank 0] PASS: {world} ranks, one per GPU, NCCL all-reduce works. "
                  f"Safe to launch FSDP training.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
