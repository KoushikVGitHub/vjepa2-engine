"""JEPA Architecture: I-JEPA vs. LeJEPA and the Mechanics of Collapse.

Goal of this file: Provide the complete, from-scratch implementation of the Joint-Embedding
Predictive Architecture (JEPA), and empirically demonstrate exactly WHAT prevents 
representation collapse.

The Collapse Problem: 
    In joint-embedding architectures, if the model trivially maps all inputs to the same 
    constant vector, the prediction loss drops to 0, but no semantic features are learned. 
    A collapsing model is detected when the standard deviation of the target embeddings 
    (`tgt_std`) crashes to 0 alongside the loss.

This module implements two distinct anti-collapse mechanisms behind the `loss_mode` switch:

Mode 1: "ema" (The Asymmetric Baseline, e.g., I-JEPA, SimSiam)
    Topology:
        context (masked) patches --> context_encoder --> predictor --(+ target pos)--> pred
        full image ---------------> target_encoder (EMA, stop-grad) -[target_idx]--> tgt
        loss = smooth_L1(pred, tgt)
    Mechanism: 
        The STOP-GRADIENT on the target branch is the load-bearing mechanism that prevents 
        collapse. The EMA target encoder simply provides a stable, slowly-evolving teacher. 
        If you remove the stop-grad in this setup, the model collapses immediately.

Mode 2: "lejepa" (The Symmetric Architecture, Balestriero & LeCun 2025)
    Topology:
        context (masked) patches --> shared_encoder ---> predictor --(+ target pos)--> pred
        full image ---------------> shared_encoder (WITH GRAD) -----[target_idx]--> tgt
        loss = smooth_L1(pred, tgt) + lambda * SIGReg(full_embeddings)
    Mechanism: 
        Shared encoder, NO stop-gradient, NO EMA teacher. Gradients flow directly into the 
        target branch. Normally, this topology guarantees collapse. Here, it is rescued 
        entirely by Sketched Isotropic Gaussian Regularization (SIGReg). SIGReg analytically 
        prevents collapse by forcing the entire latent distribution toward a maximum-entropy 
        Gaussian N(0, I).

The `__main__` block runs a synthetic empirical test to prove that both the stop-gradient 
and SIGReg successfully keep `tgt_std` healthy, while a symmetric model without SIGReg crashes.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from sigreg import sigreg_loss   # LeJEPA anti-collapse regularizer (see src/sigreg.py, Day-2 star)


# --------------------------------------------------------------------------- EMA + masking
@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, decay: float = 0.998):
    """
    Performs an Exponential Moving Average (EMA) update on the target network parameters.

    Behavior:
        Slowly updates the weights of the target network (`target`) to track the 
        weights of the actively training online network (`online`). Formula: 
        theta_target <- decay * theta_target + (1 - decay) * theta_online.
        
    Role in Program:
        Acts as a quality and stability booster for the standard JEPA mode. By lagging 
        behind the online network, it provides a more stable, slowly-evolving training 
        signal (teacher) for the predictor to match.
    """
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(decay).add_(po, alpha=1 - decay)


def random_block_mask(grid: int, block: int, device):
    """
    Generates a spatial block mask to partition an image into context and target regions.

    Behavior:
        Selects a random contiguous square block (size `block` x `block`) on a 2D patch 
        grid (size `grid` x `grid`). The indices falling inside the block become the 
        target tokens; everything else becomes the context tokens.

    Role in Program:
        Provides the token-level masking indices required to force the network to 
        predict missing regions (I-JEPA style) rather than trivially copying inputs.
    """

    top = torch.randint(0, grid - block + 1, (1,)).item()
    left = torch.randint(0, grid - block + 1, (1,)).item()
    all_idx = torch.arange(grid * grid, device=device)
    is_target = torch.zeros(grid * grid, dtype=torch.bool, device=device)
    for r in range(top, top + block):
        for c in range(left, left + block):
            is_target[r * grid + c] = True
    target_idx = all_idx[is_target]
    context_idx = all_idx[~is_target]
    return context_idx, target_idx


# --------------------------------------------------------------------------- tiny ViT pieces
class ViTEncoder(nn.Module):
    """
    A minimal Vision Transformer (ViT) encoder for patchified images.

    Behavior:
        Transforms an image into non-overlapping patches, projects them into a latent 
        space, adds learned positional embeddings, and processes them through a series 
        of transformer encoder layers. When the `keep` argument is used in the forward 
        pass, it processes ONLY the specified tokens (masked forward).

    Role in Program:
        Serves as the feature extractor (both the context encoder and, optionally, 
        the target encoder) mapped to the image space.
    """

    def __init__(self, img=16, patch=4, d=64, heads=4, layers=2):
        super().__init__()
        self.grid = img // patch
        self.n = self.grid ** 2
        self.proj = nn.Linear(patch * patch, d)
        self.pos = nn.Parameter(torch.randn(self.n, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.patch = patch

    def patchify(self, x):  # (B,1,H,W) -> (B, n, patch*patch)
        B = x.size(0)
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)            # (B,1,grid,grid,p,p)
        x = x.contiguous().view(B, self.n, p * p)
        return x

    def forward(self, x, keep=None):
        tok = self.proj(self.patchify(x)) + self.pos     # (B, n, d)
        if keep is not None:
            tok = tok[:, keep]                           # context = subset (keeps its pos)
        return self.blocks(tok)


class ViTPredictor(nn.Module):
    """I-JEPA-style predictor: a LIGHTER transformer that predicts target latents from context.

    Projects context tokens from the encoder dim (d) down to a narrower predictor dim (pred_d),
    runs a few blocks, then projects back to d so the prediction matches the target latents
    (which live in encoder-dim space). Keeping the predictor light is the I-JEPA default -- most
    capacity should sit in the ENCODER, which is the part the probe actually uses.
    """

    def __init__(self, n, d, pred_d=384, heads=6, layers=6):
        super().__init__()
        self.embed = nn.Linear(d, pred_d)                            # encoder dim -> predictor dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, pred_d) * 0.02)
        self.pos = nn.Parameter(torch.randn(n, pred_d) * 0.02)
        layer = nn.TransformerEncoderLayer(pred_d, heads, pred_d * 2, batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.proj = nn.Linear(pred_d, d)                             # predictor dim -> encoder dim

    def forward(self, ctx, ctx_idx, target_idx):
        B = ctx.size(0)
        ctx = self.embed(ctx) + self.pos[ctx_idx]                    # (B, n_ctx, pred_d), + ctx pos
        n_ctx = ctx.size(1)
        masks = self.mask_token.expand(B, len(target_idx), -1) + self.pos[target_idx]
        x = torch.cat([ctx, masks], dim=1)
        x = self.blocks(x)
        return self.proj(x[:, n_ctx:])                               # -> encoder dim, matches target


class JEPA(nn.Module):
    """
    The orchestrating module for the Joint-Embedding Predictive Architecture.

    Behavior:
        Manages the interplay between the context encoder, target encoder (if used), 
        and the predictor. Depending on `loss_mode`, it either uses a traditional EMA 
        teacher with a stop-gradient (`"ema"`) or a single shared encoder relying on 
        SIGReg regularization to prevent collapse (`"lejepa"`).

    Role in Program:
        The central container testing the structural hypotheses of representation collapse.
    """

    def __init__(self, encoder, predictor, ema_decay=0.998, stop_grad=True,
                 loss_mode="ema", sigreg_lambda=0.02):
        super().__init__()
        self.context_encoder = encoder
        self.predictor = predictor
        self.ema_decay = ema_decay
        self.stop_grad = stop_grad                                    # ema-mode: the load-bearing switch
        self.loss_mode = loss_mode                                    # "ema" | "lejepa"
        self.sigreg_lambda = sigreg_lambda                            # LeJEPA reg weight (paper: 0.02)

        # EMA teacher exists ONLY in ema mode. In lejepa there is no teacher (SIGReg replaces
        # stop-grad), so skip the deepcopy -> saves ~a whole encoder's params of dead memory
        # per GPU (matters at ViT-L). [plumbing done for you; the ML lives in _forward_lejepa]
        if loss_mode == "ema":
            self.target_encoder = copy.deepcopy(encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)                               # stop-grad on target
        else:
            self.target_encoder = None

    def forward(self, x, context_idx, target_idx,
                sigreg_generator=None, sigreg_distributed=False):
        # Dispatch on the anti-collapse mechanism. The sigreg_* kwargs are used only by lejepa
        # (the training loop supplies a per-step, rank-synced generator + the distributed flag).
        if self.loss_mode == "ema":
            return self._forward_ema(x, context_idx, target_idx)
        elif self.loss_mode == "lejepa":
            return self._forward_lejepa(x, context_idx, target_idx,
                                        sigreg_generator, sigreg_distributed)
        raise ValueError(f"unknown loss_mode {self.loss_mode!r} (expected 'ema' | 'lejepa').")

    def _forward_ema(self, x, context_idx, target_idx):
        "Executes the standard JEPA forward pass using an EMA teacher and stop-gradient."

        ctx = self.context_encoder(x, keep=context_idx)               # online (has grad)
        if self.stop_grad:
            with torch.no_grad():
                tgt = self.target_encoder(x)[:, target_idx]          # EMA target, NO grad
        else:
            # symmetric / no-stop-grad: target is the SAME trainable encoder, grad flows in.
            # Now the global minimum is a constant vector -> collapse. This is the control.
            tgt = self.context_encoder(x)[:, target_idx]
        pred = self.predictor(ctx, context_idx, target_idx)
        # When stop_grad=False, tgt carries grad -> the loss can collapse BOTH sides to a constant.
        loss = F.smooth_l1_loss(pred, tgt)                           # latent-space loss
        return loss, tgt

    def _forward_lejepa(self, x, context_idx, target_idx,
                        sigreg_generator=None, sigreg_distributed=False):
        "Executes the LeJEPA forward pass using a shared encoder and SIGReg regularization."

        # (1) Context tokens for the predictor -- the MASKED forward (keep=context_idx).
        ctx = self.context_encoder(x, keep=context_idx)

        # (2) Full-image encoding, WITH grad -> feeds both the targets (3) and SIGReg (6).
        full = self.context_encoder(x)                        # (B, n, d)

        # (3) Target latents = target positions of the full encoding. NO stop-grad; grad flows
        #     into tgt (unlike ema mode). SIGReg is what stops the collapse this would cause.
        tgt = full[:, target_idx]

        # (4) Predict target latents from context.
        pred = self.predictor(ctx, context_idx, target_idx)

        # (5) Prediction loss in latent space.
        pred_loss = F.smooth_l1_loss(pred, tgt)

        # (6) SIGReg over ALL token embeddings: flatten (B, n, d) -> (B*n, d), regularize toward
        #     N(0, I). In distributed training the loop passes a rank-synced generator + the
        #     distributed flag so the ECF sums all-reduce into a global-batch statistic.
        reg = sigreg_loss(full.reshape(-1, full.size(-1)),
                          generator=sigreg_generator, distributed=sigreg_distributed)

        # (7) Combine with the LeJEPA weight.
        loss = (1 - self.sigreg_lambda) * pred_loss + self.sigreg_lambda * reg

        # (optional) stash components for logging / collapse watch:
        self.last_pred = pred_loss.item()
        self.last_reg = reg.item()
        self.last_tgt_std = tgt.detach().float().std(dim=0).mean().item()   # COMPLETE-collapse detector

        # Effective rank (participation ratio) of the target embeddings -> catches DIMENSIONAL
        # collapse: variance stays healthy but piles onto a few dims, which tgt_std alone misses
        # (and which often drops BEFORE tgt_std craters). PR = tr(C)^2 / ||C||_F^2 in [1, d];
        # ~d = full-rank isotropic (healthy), -> 1 = rank-1 collapse. No eigendecomposition needed.
        with torch.no_grad():
            z = tgt.detach().reshape(-1, tgt.size(-1)).float()
            z = z - z.mean(dim=0, keepdim=True)
            C = (z.t() @ z) / z.size(0)                     # (d, d) covariance, cheap at d=1024
            tr = torch.diagonal(C).sum()
            self.last_eff_rank = (tr * tr / (C.pow(2).sum() + 1e-12)).item()

        return loss, tgt

    def step_ema(self):
        "Triggers the EMA parameter update for the target network. No-op in LeJEPA mode."
        if self.loss_mode == "ema":
            ema_update(self.target_encoder, self.context_encoder, self.ema_decay)


# --------------------------------------------------------------------------- toy data + train
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
