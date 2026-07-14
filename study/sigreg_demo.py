"""SIGReg sanity demo (pedagogical): the objective should be ~0 for a true isotropic Gaussian
and LARGE for collapsed or wrongly-scaled batches -- the intuition behind the LeJEPA regularizer.

Imports the production `sigreg_loss` from ../src. For the DISTRIBUTED correctness gate (that the
all-reduced multi-GPU statistic equals the single-device one) see `verify_all_reducible` in
src/sigreg.py, run via:  torchrun --standalone --nproc_per_node=2 src/sigreg.py --verify

Run:  python study/sigreg_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from sigreg import sigreg_loss


if __name__ == "__main__":
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
