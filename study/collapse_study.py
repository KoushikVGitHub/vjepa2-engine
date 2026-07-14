"""Day-2 study: WHAT actually prevents JEPA representation collapse?

Standalone pedagogical experiment on SYNTHETIC data that isolates the anti-collapse mechanism
behind the `loss_mode` switch. It imports the production model/loss library from ../src; nothing
here is imported by the CAMELS engine, so the library files stay free of toy-training code.

Hypothesis under test: it's the STOP-GRADIENT, not the EMA decay, that carries JEPA (SimSiam,
2020) -- and that SIGReg (LeJEPA) can REPLACE the stop-grad entirely.

Run:  python study/collapse_study.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F

from jepa_loss import JEPA, ViTEncoder, ViTPredictor, random_block_mask


def make_batch(B, img=16, device="cpu"):
    """
    Generates synthetic, spatially-correlated image batches for training tests.

    Behavior:
        Creates a tensor of smoothed random noise, normalized to unit variance.
        The smoothing ensures that local patches have structural similarity to their
        neighbors, making the context predictive of the targets (essential for a
        JEPA to actually learn rather than randomly guess).
    """
    x = torch.randn(B, 1, img, img, device=device)
    x = F.avg_pool2d(x, 5, stride=1, padding=2)                       # blur -> local structure
    return (x - x.mean()) / (x.std() + 1e-6)


def train(decay=0.998, stop_grad=True, loss_mode="ema", sigreg_lambda=0.02,
          steps=300, B=128, img=16, patch=4, d=64, seed=0, device="cpu"):
    """
    A minimal training loop designed to empirically test representation collapse.

    Behavior:
        Initializes the model architecture under specific configurations (EMA settings,
        stop-gradient flags, loss modes). It loops through dummy batches, computing
        losses and optimizing the networks. It tracks the standard deviation (`tgt_std`)
        of the target embeddings.

    Role in Program:
        The empirical testbed that falsifies the 'EMA prevents collapse' hypothesis and
        proves that either a stop-gradient (JEPA) or an explicit regularization term
        like SIGReg (LeJEPA) is necessary to keep embeddings from collapsing to a constant.
    """

    torch.manual_seed(seed)
    enc = ViTEncoder(img, patch, d)
    grid = img // patch
    pred = ViTPredictor(grid * grid, d=d, pred_d=d, heads=4, layers=2)
    model = JEPA(enc, pred, ema_decay=decay, stop_grad=stop_grad,
                 loss_mode=loss_mode, sigreg_lambda=sigreg_lambda).to(device)
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=1e-3)

    traj = []
    for step in range(steps):
        x = make_batch(B, img, device)
        ctx_idx, tgt_idx = random_block_mask(grid, block=2, device=device)
        loss, tgt = model(x, ctx_idx, tgt_idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.step_ema()
        if step % 50 == 0 or step == steps - 1:
            std = tgt.detach().std(dim=0).mean().item()               # collapse detector
            traj.append((step, loss.item(), std))
    return traj


if __name__ == "__main__":
    print("Day-2 build: WHAT actually prevents collapse? (corrected after the first run)\n")
    print("Hypothesis under test: it's the STOP-GRADIENT, not the EMA decay (SimSiam, 2020).\n")
    configs = [
        ("no stop-grad (symmetric)",   dict(decay=0.998, stop_grad=False)),
        ("stop-grad, decay=0.0",       dict(decay=0.0,   stop_grad=True)),
        ("stop-grad, decay=0.998",     dict(decay=0.998, stop_grad=True)),
        ("lejepa (SIGReg, grad->tgt)", dict(loss_mode="lejepa")),
    ]
    print(f"{'config':>26} | {'step':>4} {'loss':>9} {'tgt_std':>9}")
    for name, kw in configs:
        traj = train(**kw)
        for (step, loss, std) in traj:
            tag = ""
            if step == traj[-1][0]:
                tag = "  <- COLLAPSED" if std < 0.05 else "  <- healthy"
            print(f"{name:>26} | {step:>4} {loss:>9.5f} {std:>9.5f}{tag}")
        print("-" * 64)
    print("\nReading: removing the stop-grad (symmetric) -> std->0 & loss->0 == COLLAPSE.")
    print("         WITH stop-grad, even decay=0.0 stays healthy -> EMA is a stabilizer,")
    print("         the stop-gradient (+predictor) is the load-bearing anti-collapse piece.")
    print("         lejepa: SAME grad-into-target topology, but SIGReg keeps std healthy ->")
    print("         SIGReg REPLACES stop-grad as the anti-collapse mechanism (LeJEPA).")
