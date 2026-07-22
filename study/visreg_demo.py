"""VISReg demo (pedagogical): the crux claim behind replacing SIGReg with VISReg.

Both objectives are ~0 for a true isotropic Gaussian and large for a collapsed batch. The
DIFFERENCE that matters is the GRADIENT under *anisotropic* (dimensional) collapse -- a rank-2 blob
that still holds full unit total variance. SIGReg's characteristic-function test is nearly blind
there (its gradient vanishes -- the exact failure that forced the VICReg var/cov patch in this
repo); VISReg's sorted-quantile (sliced-Wasserstein) shape term keeps a strong gradient, so it
needs no patch.

Imports the production `visreg_loss` and `sigreg_loss` from ../src.  Run:  python study/visreg_demo.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from visreg import visreg_loss
from sigreg import sigreg_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 8192, 64

    gaussian = torch.randn(N, D)                              # target: ~0 loss
    collapsed = torch.zeros(N, D) + 0.01 * torch.randn(1, D)  # complete collapse
    # Anisotropic (dimensional) collapse: full unit TOTAL variance, but packed into 2 of D dims.
    aniso = torch.zeros(N, D)
    aniso[:, :2] = torch.randn(N, 2) * math.sqrt(D / 2)       # same total variance, rank 2

    print(f"{'batch':22s} {'VISReg':>10s} {'SIGReg':>10s}")
    for name, x in [("N(0,I) target", gaussian), ("complete-collapse", collapsed),
                    ("anisotropic rank-2", aniso)]:
        print(f"{name:22s} {visreg_loss(x).item():10.5f} {sigreg_loss(x).item():10.5f}")

    # The headline: gradient magnitude AT the anisotropic-collapse state.
    for name, fn in [("VISReg", visreg_loss), ("SIGReg", sigreg_loss)]:
        x = aniso.clone().requires_grad_(True)
        fn(x).backward()
        print(f"grad-norm @ rank-2 collapse | {name:8s} = {x.grad.norm().item():.4e}")
    print("Expect VISReg's rank-2 grad-norm >> SIGReg's (the reason it replaces the var/cov patch).")
