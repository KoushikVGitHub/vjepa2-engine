"""Distributed SIGReg correctness: world=2 x batch-B must equal world=1 x batch-2B.

This is the property that makes SIGReg worth using at scale -- its regularizer is an
expectation over the batch, so ranks all-reduce partial ECF sums instead of gathering
negative pairs across the wire (contrast SimCLR). If the all-reduce were non-differentiable,
or the sums were reduced AFTER the nonlinearity instead of before, training would still run
and still look plausible -- it would just be silently optimizing the wrong objective.

`src/sigreg.py --verify` gates this on 2 GPUs over NCCL. That cannot run in CI, so this is
the same invariant over GLOO on CPU: same claim, no hardware, every push.
"""
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

B, D, SEED = 256, 32, 0
WORLD = 2


def _worker(rank, world, port, out):
    """One rank: run distributed SIGReg on its shard, report loss + local grad."""
    sys.path.insert(0, REPO_SRC)
    from sigreg import sigreg_loss

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(SEED)
        full = torch.randn(world * B, D)                 # identical on every rank
        shard = full[rank * B:(rank + 1) * B].detach().requires_grad_()

        # Rank-SYNCED generator: V and t must match across ranks or the all-reduced
        # statistic mixes different projections and means nothing.
        gen = torch.Generator().manual_seed(SEED)
        loss = sigreg_loss(shard, generator=gen, distributed=True)
        loss.backward()
        out[rank] = (loss.item(), shard.grad.clone())
    finally:
        dist.destroy_process_group()


def _reference():
    """The single-process answer on the FULL batch -- what sharding must reproduce."""
    sys.path.insert(0, REPO_SRC)
    from sigreg import sigreg_loss

    torch.manual_seed(SEED)
    full = torch.randn(WORLD * B, D).detach().requires_grad_()
    gen = torch.Generator().manual_seed(SEED)
    loss = sigreg_loss(full, generator=gen, distributed=False)
    loss.backward()
    return loss.item(), full.grad


def test_ecf_sums_are_additive_across_shards(gen):
    """The reason the all-reduce is legal, checked WITHOUT a process group.

    SIGReg reduces SUMS and divides by the global N; the nonlinearity is applied after. So
    concatenating shards must give the same ECF as summing their partial sums. This is the
    algebra the collective relies on -- gated here so it also runs where gloo cannot (Windows),
    and so a regression points at the math rather than at the networking.
    """
    sys.path.insert(0, REPO_SRC)
    from sigreg import random_directions

    torch.manual_seed(SEED)
    full = torch.randn(WORLD * B, D)
    V = random_directions(D, 16, full.device, generator=gen)
    t = torch.randn(8, generator=gen)

    def ecf_sums(z):
        tp = (z @ V).unsqueeze(-1) * t
        return tp.cos().sum(dim=0), tp.sin().sum(dim=0)

    cos_full, sin_full = ecf_sums(full)
    shards = [ecf_sums(full[r * B:(r + 1) * B]) for r in range(WORLD)]
    cos_parts = sum(s[0] for s in shards)
    sin_parts = sum(s[1] for s in shards)

    assert torch.allclose(cos_full, cos_parts, atol=1e-4)
    assert torch.allclose(sin_full, sin_parts, atol=1e-4)


@pytest.mark.skipif(sys.platform == "win32",
                    reason="gloo cannot create a device on Windows (makeDeviceForHostname)")
@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend unavailable")
def test_sharded_sigreg_matches_the_full_batch():
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    mp.start_processes(_worker, args=(WORLD, 29511, out), nprocs=WORLD,
                       start_method="spawn", join=True)
    assert len(out) == WORLD, "a rank died before reporting"

    ref_loss, ref_grad = _reference()

    # (1) Every rank computes the SAME global-batch loss, equal to the unsharded reference.
    for rank in range(WORLD):
        got = out[rank][0]
        assert abs(got - ref_loss) < 1e-5, (
            f"rank {rank} loss {got:.8f} != single-process reference {ref_loss:.8f}"
        )

    # (2) Each rank's gradient matches its slice of the reference gradient. The differentiable
    #     all-reduce sums grads on the way back, so each rank sees the reference slice scaled by
    #     world -- the same factor DDP/FSDP then averages out. Assert the exact relationship
    #     rather than "close enough either way", so a change in it is a test failure, not a shrug.
    for rank in range(WORLD):
        grad = out[rank][1]
        expect = ref_grad[rank * B:(rank + 1) * B] * WORLD
        assert torch.allclose(grad, expect, rtol=1e-4, atol=1e-6), (
            f"rank {rank}: sharded gradient != world x reference slice "
            f"(max diff {(grad - expect).abs().max().item():.3e})"
        )
