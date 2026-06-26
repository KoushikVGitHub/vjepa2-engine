"""Day 4 — toy-scale distributed training (PyTorch FSDP/DDP, bf16).

Maps to AMI's preferred quals: large-scale distributed training + efficiency.
Run: torchrun --nproc_per_node=2 src/train_fsdp.py
Report throughput before/after FSDP + bf16 + activation checkpointing.
"""
import os
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def main():
    local_rank = setup()

    # TODO(Day4): build the JEPA model from src/jepa_loss.py
    model = ...  # nn.Module
    model = FSDP(
        model,
        # TODO(Day4): auto_wrap_policy for transformer blocks,
        # mixed_precision=bf16 policy, activation checkpointing.
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

    # TODO(Day4): build distributed dataloader (DistributedSampler) from data/curation.py
    loader = ...

    model.train()
    t0 = time.time(); seen = 0
    for step, batch in enumerate(loader):
        batch = batch.cuda(local_rank, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)          # JEPA returns latent-space loss
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        # model.step_ema()  # update EMA target
        seen += batch.shape[0]
        if step % 20 == 0 and local_rank == 0:
            print(f"step {step} loss {loss.item():.4f} "
                  f"{seen/(time.time()-t0):.1f} samples/sec")
        if step >= 200:
            break

    cleanup()


if __name__ == "__main__":
    main()
