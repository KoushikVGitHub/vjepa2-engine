"""SIGReg: Sketched Isotropic Gaussian Regularization.

Goal of this file: Implement the mathematical engine of LeJEPA (Balestriero & LeCun, 2025). 
This module replaces standard anti-collapse heuristics (like EMA teachers and stop-gradients) 
with a single, principled statistical objective: explicitly forcing the embedding distribution 
to match an isotropic Gaussian N(0, I).

The Anti-Collapse Mechanism:
    An isotropic Gaussian represents the maximum-entropy distribution for a given variance, 
    meaning it carries maximal information. Conversely, a collapsed representation (where all 
    inputs map to a constant or low-rank subspace) is maximally non-Gaussian. By explicitly 
    penalizing distance from N(0, I), the regularizer naturally repels collapse.

Mathematical Mechanics:
    1. Cramer-Wold Theorem (The "Sketch"): Proving a high-dimensional distribution is N(0, I) 
       is computationally intractable. However, Cramer-Wold states a distribution is N(0, I) 
       if and only if *every* 1-D projection is N(0, 1). We approximate this by drawing 
       random unit vectors (the sketch) and projecting the embeddings into 1-D.
    2. Characteristic Function (The Test): We test the 1-D projections for normality using 
       their empirical characteristic function (CF). We compute the mean squared difference 
       between the empirical CF of our batch and the theoretical CF of a standard normal 
       distribution (exp(-t^2/2)) across sampled frequencies.

Distributed Scaling:
    Because the CF calculation is a pure expectation (mean) over the batch, scaling to 
    multi-GPU (FSDP/DDP) is highly efficient. Ranks simply all-reduce (sum) their local 
    CF components to compute the global statistic. There is no need to gather or broadcast 
    massive batches of negative pairs across the network.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
import os
import sys

def random_directions(dim: int, n_proj: int, device, generator=None) -> torch.Tensor:
    """
    Generates the mathematical "sketch" used to reduce the high-dimensional problem 
    into multiple independent 1-D problems.
    
    Behavior: 
        Creates a tensor of random values of shape `(dim, n_proj)` and normalizes 
        them along the first dimension. This results in `n_proj` random unit vectors 
        in R^dim, which act as the projection directions for the Cramer-Wold test.
        
    Role in Program: 
        Provides the random axes onto which the high-dimensional embedding batch will 
        be projected before testing for normality.
    """
    v = torch.randn(dim, n_proj, device=device, generator=generator)
    return F.normalize(v, dim=0)


def _differentiable_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Safely aggregates partial sums across multiple GPUs while preserving PyTorch's 
    computational graph (autograd).
    
    Behavior:
        Wraps `torch.distributed.nn.functional.all_reduce` using the SUM operation. 
        Unlike standard in-place collective operations (which sever the autograd graph), 
        this function allows backpropagation to flow through the distributed reduction.
        
    Role in Program:
        Enables SIGReg to compute a true global-batch characteristic function in 
        distributed training setups, ensuring gradients properly flow back to the local 
        embeddings on each specific rank.
    """
    return all_reduce(x, op=dist.ReduceOp.SUM)


def sigreg_loss(z: torch.Tensor, n_proj: int = 256, n_freq: int = 64,
                generator: torch.Generator = None, distributed: bool = False) -> torch.Tensor:
    """
    The core objective function calculating how far the empirical distribution of 
    the batch embeddings `z` deviates from an isotropic Gaussian N(0, I).
    
    Behavior:
        1. Projects D-dimensional embeddings `z` into 1-D using random unit vectors.
        2. Samples random frequencies `t` for the characteristic function (CF).
        3. Computes the empirical CF (real and imaginary) of the 1-D projections.
        4. In distributed mode, sums CF components across all ranks via differentiable 
           all-reduce before dividing by the global batch size.
        5. Returns the mean squared difference between the empirical CF and the 
           theoretical CF of a standard normal distribution.
           
    Role in Program:
        Replaces heuristic EMA-teacher and stop-gradient mechanisms by explicitly 
        regularizing the latent space to prevent collapse.

    Args:
        z: (B, D) batch of embeddings (NOT pre-standardized).
        n_proj: number of random 1-D projections (the sketch dimension).
        n_freq: number of CF frequencies sampled per projection.
        generator: RNG for V and t. In DISTRIBUTED mode this MUST be seeded IDENTICALLY on
            every rank (e.g. seed = base + step) so all ranks project onto the SAME directions.
        distributed: if True, all-reduce the ECF SUMS across ranks to form the GLOBAL-batch
            statistic before computing the (nonlinear) loss.

    Returns:
        Scalar loss; 0 iff the projected embeddings are exactly N(0, 1) in every direction.
    """
    B, D = z.shape
    z = z.float()          # SIGReg stats in fp32: avoids the FSDP-bf16 dtype clash + precision loss
    V = random_directions(D, n_proj, z.device, generator=generator)   # (D, n_proj)
    P = z @ V                                          # (B, n_proj)  projected samples

    # Frequencies sampled from N(0,1): Monte-Carlo of the Gaussian-weighted CF integral.
    t = torch.randn(n_freq, device=z.device, generator=generator)     # (n_freq,)
    tp = P.unsqueeze(-1) * t                           # (B, n_proj, n_freq)

    # Reduce STATS, not the loss.
    cos_sum = tp.cos().sum(dim=0)                      # (n_proj, n_freq)
    sin_sum = tp.sin().sum(dim=0)

    if distributed:
        assert generator is not None, \
            "distributed SIGReg needs a rank-SYNCED generator (V and t must match across ranks)"
        world = dist.get_world_size()

        # Differentiable SUM all-reduce so grad flows back to each rank's local samples.
        cos_sum = _differentiable_all_reduce_sum(cos_sum)
        sin_sum = _differentiable_all_reduce_sum(sin_sum)
        N = B * world
    else:
        N = B

    cos = cos_sum / N                                  # global empirical CF, real part
    sin = sin_sum / N                                  # global empirical CF, imag part

    target = torch.exp(-0.5 * t**2)                    # CF of N(0,1): real, (n_freq,)

    # Squared CF distance: real part should match exp(-t^2/2), imaginary part should be 0.
    loss = ((cos - target) ** 2 + sin ** 2).mean()
    return loss


def verify_all_reducible(B: int = 512, D: int = 64, seed: int = 0):
    """
    A diagnostic testing utility to mathematically prove that the distributed 
    implementation of `sigreg_loss` is strictly equivalent to a single-device implementation.

    Run distributed:   torchrun --nproc_per_node=2 src/sigreg.py --verify

    Behavior/Invariant: 
        SIGReg on a global batch must not depend on how the batch is split across ranks. 
        It builds a global reference batch, splits it, runs distributed loss/backward on 
        the shard, and compares it to a full reference pass on Rank 0. It asserts both 
        loss values and gradients match (accounting for the expected `world_size` scaling).
        
    Role in Program:
        Catches scaling bugs or broken autograd graphs in isolation before they are 
        hidden away inside a massive multi-node FSDP training loop.
    """
    # 1. Initialize process group and read environment variables
    dist.init_process_group("nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))


    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 2. Build the FULL reference batch from a seeded generator
    torch.manual_seed(seed)
    full = torch.randn(world * B, D)

    # Isolate this specific rank's shard and move it to the correct GPU
    shard = full[rank * B: (rank + 1) * B].to(device).detach().requires_grad_()

    # 3. Compute distributed loss and backward pass
    sync_gen = torch.Generator(device).manual_seed(seed)   # identical on all ranks -> V,t match
    l_dist = sigreg_loss(shard, generator=sync_gen, distributed=True)
    l_dist.backward()

    # 4. On Rank 0, compute the non-distributed reference and compare
    if rank == 0:
        # Prepare the full tensor for the reference pass
        full_ref = full.to(device).detach().requires_grad_()
        sync_gen_ref = torch.Generator(device).manual_seed(seed)

        # Calculate reference loss (distributed=False)
        l_ref = sigreg_loss(full_ref, generator=sync_gen_ref, distributed=False)
        l_ref.backward()

        # 5. Assertions
        print(f"Distributed Loss: {l_dist.item():.6f} | Reference Loss: {l_ref.item():.6f}")
        
        # Check Loss
        assert torch.allclose(l_dist, l_ref, rtol=1e-5, atol=1e-5), "Loss mismatch between distributed and reference!"
        
        # Slice the reference gradient to compare against Rank 0's shard gradient
        ref_grad_slice = full_ref.grad[0:B]

        # Check Gradients (watching for the expected xW factor)
        if torch.allclose(shard.grad, ref_grad_slice, rtol=1e-5, atol=1e-5):
            print("Success: Gradients match perfectly!")
        elif torch.allclose(shard.grad, ref_grad_slice * world, rtol=1e-5, atol=1e-5):
            print(f"Warning: Gradients match, but are scaled by world_size (x{world}). Expected DDP/FSDP interaction detected.")
        else:
            raise AssertionError("Gradient mismatch: Distributed gradients do not match the reference!")
        
    # Cleanup to prevent hanging processes
    dist.destroy_process_group()


    


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify_all_reducible()
        raise SystemExit(0)

    # Sanity demo: SIGReg should be ~0 for true N(0,I), and LARGE for collapsed/structured data.
    torch.manual_seed(0)
    B, D = 4096, 64

    gaussian = torch.randn(B, D)                       # the target distribution
    collapsed = torch.zeros(B, D) + 0.01 * torch.randn(1, D)  # near-constant == collapse
    scaled = 5.0 * torch.randn(B, D)                   # right shape, wrong variance
    uniform = (torch.rand(B, D) - 0.5) * 3.46          # zero-mean, ~unit-var, but NOT Gaussian

    for name, x in [("N(0,I) target", gaussian),
                    ("collapsed", collapsed),
                    ("scaled var=25", scaled),
                    ("uniform", uniform)]:
        print(f"{name:18s} SIGReg = {sigreg_loss(x).item():.5f}")
    # Expected: target ~0; collapsed and scaled large; uniform small-but-nonzero.
