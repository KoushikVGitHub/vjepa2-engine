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

from sigreg import sigreg_loss, variance_covariance_reg   # LeJEPA reg + VICReg anisotropy patch
from visreg import visreg_loss                            # VISReg = successor single-objective reg


# --------------------------------------------------------------------------- loss-mode registry
# Single source of truth for what each anti-collapse mode NEEDS. Every mode-dependent branch --
# here (teacher allocation, EMA no-op) and in the training harness (MFU accounting, the collapse
# guard, per-step metric logging, the synced SIGReg generator, and the argparse --loss choices) --
# reads these flags instead of comparing `loss_mode == "..."`. Adding a 4th mode is then: add one
# row + write its `_forward_<mode>` method; no scattered string comparisons to hunt down.
#
#   needs_teacher : allocate an EMA target encoder + stop-grad (ema only). Also drives the MFU
#                   FLOP model (frozen half-size teacher forward vs a 2nd full online forward).
#   regularized   : uses a distributional anti-collapse reg -> emit pred/reg/tgt_std/eff_rank
#                   metrics and arm the collapse-abort guard (lejepa, visreg).
#   synced_gen    : the reg is a global-batch statistic needing a rank-SYNCED generator +
#                   differentiable all-reduce (SIGReg/lejepa). VISReg is a local-shard stat (its
#                   sort does not cleanly all-reduce), so it needs neither.
LOSS_MODES = {
    "ema":    {"needs_teacher": True,  "regularized": False, "synced_gen": False},
    "lejepa": {"needs_teacher": False, "regularized": True,  "synced_gen": True},
    "visreg": {"needs_teacher": False, "regularized": True,  "synced_gen": False},
}


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


def random_block_mask(grid: int, block: int, device, n_blocks: int = 1):
    """
    Generates a spatial block mask to partition an image into context and target regions.

    Behavior:
        Selects `n_blocks` random contiguous square blocks (each `block` x `block`) on a 2D
        patch grid (size `grid` x `grid`). Their UNION becomes the target tokens; everything
        else is context. Overlaps are handled naturally by the boolean OR, so the effective
        target ratio is at most n_blocks*block^2 / grid^2 (less when blocks overlap).

    Role in Program:
        Forces the network to predict missing regions (I-JEPA style). A SINGLE small block is
        a trivially easy task (a low-rank cheat suffices, feeding dimensional collapse); several
        blocks covering ~15-25% make the prediction hard enough to demand richer features.
    """

    all_idx = torch.arange(grid * grid, device=device)
    is_target = torch.zeros(grid * grid, dtype=torch.bool, device=device)
    for _ in range(n_blocks):
        top = torch.randint(0, grid - block + 1, (1,)).item()
        left = torch.randint(0, grid - block + 1, (1,)).item()
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
                 loss_mode="ema", sigreg_lambda=0.02,
                 var_coef=0.0, cov_coef=0.0, target_norm=False,
                 visreg_coef=1.0):
        super().__init__()
        if loss_mode not in LOSS_MODES:
            raise ValueError(f"unknown loss_mode {loss_mode!r} (expected one of {list(LOSS_MODES)}).")
        self.context_encoder = encoder
        self.predictor = predictor
        self.ema_decay = ema_decay
        self.stop_grad = stop_grad                                    # ema-mode: the load-bearing switch
        self.loss_mode = loss_mode                                    # see LOSS_MODES registry
        self.sigreg_lambda = sigreg_lambda                            # LeJEPA reg weight (paper: 0.02)
        # VISReg weight on its single scale+shape+center regularizer. Unlike SIGReg (balanced via the
        # (1-lambda)/lambda convex combo) VISReg is added straight: loss = pred + visreg_coef * reg
        # (paper weights the three sub-terms equally at 1). Its raw magnitude is ~O(1), NOT ~SIGReg's,
        # so it does NOT share sigreg_lambda -- tune this independently.
        self.visreg_coef = visreg_coef
        # Anisotropic-collapse patch (lejepa mode). SIGReg's marginal test barely resists rank
        # collapse when total variance is spread right; the VICReg var/cov term does (see sigreg.py).
        self.var_coef = var_coef                                     # variance-hinge weight (~1e-2)
        self.cov_coef = cov_coef                                     # off-diagonal decorrelation (~2e-2)
        # Normalize (no-affine LayerNorm) pred & tgt before the L1: removes the "shrink the target
        # toward a constant" shortcut that DRIVES collapse. This is the I-JEPA/V-JEPA target-norm.
        self.target_norm = target_norm

        # EMA teacher exists ONLY in ema mode. In lejepa/visreg there is no teacher (the
        # distributional regularizer replaces stop-grad), so skip the deepcopy -> saves ~a whole
        # encoder's params of dead memory per GPU (matters at ViT-L).
        if LOSS_MODES[loss_mode]["needs_teacher"]:
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
        elif self.loss_mode == "visreg":
            return self._forward_visreg(x, context_idx, target_idx)
        raise ValueError(
            f"unknown loss_mode {self.loss_mode!r} (expected 'ema' | 'lejepa' | 'visreg').")

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

        # (5) Prediction loss in latent space. When target_norm is on, no-affine LayerNorm both
        #     sides first: the L1 then measures SHAPE agreement, so shrinking tgt toward a constant
        #     no longer lowers the loss (constant -> undefined after norm) -> collapse driver removed.
        if self.target_norm:
            d = pred.size(-1)
            pred_loss = F.smooth_l1_loss(F.layer_norm(pred, (d,)), F.layer_norm(tgt, (d,)))
        else:
            pred_loss = F.smooth_l1_loss(pred, tgt)

        # (6) SIGReg over ALL token embeddings: flatten (B, n, d) -> (B*n, d), regularize toward
        #     N(0, I). In distributed training the loop passes a rank-synced generator + the
        #     distributed flag so the ECF sums all-reduce into a global-batch statistic.
        full_flat = full.reshape(-1, full.size(-1))
        reg = sigreg_loss(full_flat, generator=sigreg_generator, distributed=sigreg_distributed)

        # (6b) VICReg var-hinge + off-diagonal covariance on the SAME full embeddings. SIGReg alone
        #      barely resists anisotropic (dimensional) collapse; this term supplies the strong,
        #      correctly-directed gradient against low rank. Local-shard stats (see sigreg.py).
        if self.var_coef > 0 or self.cov_coef > 0:
            var_l, cov_l = variance_covariance_reg(full_flat)
        else:
            var_l = cov_l = torch.zeros((), device=full.device)

        # (7) Combine: LeJEPA prediction/SIGReg balance + the anisotropy patch.
        loss = ((1 - self.sigreg_lambda) * pred_loss + self.sigreg_lambda * reg
                + self.var_coef * var_l + self.cov_coef * cov_l)

        # Stash components + collapse detectors for logging / the training-loop abort guard.
        self.last_pred = pred_loss.item()
        self.last_reg = reg.item()
        self.last_var = var_l.item()
        self.last_cov = cov_l.item()
        self._stash_collapse_stats(tgt, full_flat)

        return loss, tgt

    def _forward_visreg(self, x, context_idx, target_idx):
        "VISReg forward: shared encoder + single sliced-Wasserstein-to-Gaussian reg, no teacher, no patch."

        # (1-2) Same two-forward topology as lejepa: masked context for the predictor, full-image
        #       encoding WITH grad feeding both the targets and the regularizer.
        ctx = self.context_encoder(x, keep=context_idx)
        full = self.context_encoder(x)                        # (B, n, d)
        tgt = full[:, target_idx]                             # NO stop-grad; VISReg stops collapse

        # (3) Predict target latents from context (the "Invariance" term of VIS).
        pred = self.predictor(ctx, context_idx, target_idx)
        if self.target_norm:
            d = pred.size(-1)
            pred_loss = F.smooth_l1_loss(F.layer_norm(pred, (d,)), F.layer_norm(tgt, (d,)))
        else:
            pred_loss = F.smooth_l1_loss(pred, tgt)

        # (4) VISReg over ALL token embeddings. One objective = variance(scale) + sketch(shape) +
        #     center; its non-vanishing gradient under dimensional collapse makes the SIGReg
        #     var/cov patch unnecessary (see src/visreg.py). Local-shard stat, no synced generator.
        full_flat = full.reshape(-1, full.size(-1))
        reg = visreg_loss(full_flat)

        # (5) Straight add (paper weights the sub-terms equally); no (1-lambda)/lambda convex combo.
        loss = pred_loss + self.visreg_coef * reg

        # Stash for logging / the abort guard. var/cov are not part of VISReg -> report 0 so the
        # shared log line stays uniform across modes.
        self.last_pred = pred_loss.item()
        self.last_reg = reg.item()
        self.last_var = 0.0
        self.last_cov = 0.0
        self._stash_collapse_stats(tgt, full_flat)

        return loss, tgt

    def _stash_collapse_stats(self, tgt, full_flat):
        """Record the two collapse detectors used by logging and the training-loop abort guard.

        last_tgt_std -> 0 is COMPLETE collapse. last_eff_rank (participation ratio
        PR = tr(C)^2 / ||C||_F^2, in [1, d]) catches DIMENSIONAL collapse -- variance stays healthy
        but piles onto a few dims, which tgt_std alone misses and which usually drops BEFORE tgt_std
        craters. Measured on `full` (all tokens = what the regularizer sees and the probe pools),
        the true collapse signal, not just the target subset. Shared by the lejepa + visreg paths.
        """
        self.last_tgt_std = tgt.detach().float().std(dim=0).mean().item()
        with torch.no_grad():
            z = full_flat.detach().float()
            z = z - z.mean(dim=0, keepdim=True)
            C = (z.t() @ z) / z.size(0)                     # (d, d) covariance, cheap at d=1024
            tr = torch.diagonal(C).sum()
            self.last_eff_rank = (tr * tr / (C.pow(2).sum() + 1e-12)).item()

    def step_ema(self):
        "Triggers the EMA parameter update for the target network. No-op in teacher-free modes."
        if LOSS_MODES[self.loss_mode]["needs_teacher"]:
            ema_update(self.target_encoder, self.context_encoder, self.ema_decay)


# The synthetic collapse study (make_batch + train + the config comparison) lives in
# study/collapse_study.py -- it imports the classes above. This module stays a pure library.
