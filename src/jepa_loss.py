"""Day 2 BUILD - minimal JEPA from scratch: masking + tiny ViT + predictor + EMA target.

Goal of this file: rebuild the I-JEPA forward pass by hand so the mechanics are muscle,
then EMPIRICALLY test WHAT prevents collapse. The first hypothesis ("low EMA decay ->
collapse") was FALSIFIED by the run -- with the stop-gradient present, even decay=0 stays
healthy. The corrected lesson (SimSiam, Chen & He 2020), reproduced in __main__:
    The STOP-GRADIENT (+ predictor) is the load-bearing anti-collapse mechanism.
    EMA decay is a quality/stability booster, NOT the anti-collapse guarantee.
    Remove the stop-grad (symmetric, grad into both sides) -> the global minimum is a
    constant vector -> std -> 0, loss -> 0 == real collapse.

Topology (matches the pre-test answer):
    context (masked) patches --> context_encoder --> predictor --(+ target positions)--> pred
    full image -------------->  target_encoder (EMA, stop-grad) --[target_idx]--> tgt
    loss = smooth_L1(pred, tgt)   in LATENT space
Gradient flows context_encoder -> predictor. Target branch is a stop-grad, EMA-only label.

Collapse detector: std of the target embeddings across the batch. If it -> 0 while the loss
also -> 0, the model has cheated the loss by emitting a constant (collapse), NOT learned.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- EMA + masking
@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, decay: float = 0.998):
    # theta_target <- decay*theta_target + (1-decay)*theta_online
    # decay=0 -> target becomes an EXACT copy of online (no lag) -> collapse risk.
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(decay).add_(po, alpha=1 - decay)


def random_block_mask(grid: int, block: int, device):
    """Block masking on a `grid` x `grid` patch grid (I-JEPA style: a contiguous block is
    the TARGET, everything else is CONTEXT). Returns (context_idx, target_idx) as LongTensors
    of token indices into the flattened grid.
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
class TinyEncoder(nn.Module):
    """Patchify -> linear embed (+pos) -> a couple of transformer blocks -> per-patch tokens.
    forward(x, keep=idx) processes ONLY the kept tokens (context encoder sees masked input)."""

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


class TinyPredictor(nn.Module):
    """Take context tokens + the POSITIONS of the targets; predict target latents.
    Mask tokens (learned) carry the target positional embeddings -- 'predict what's here'."""

    def __init__(self, n, d=64, heads=4, layers=2):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.proj = nn.Linear(d, d)

    def forward(self, ctx, ctx_idx, target_idx):
        B, n_ctx, d = ctx.size(0), ctx.size(1), ctx.size(2)
        ctx = ctx + self.pos[ctx_idx]                                # remind predictor where ctx is
        masks = self.mask_token.expand(B, len(target_idx), d) + self.pos[target_idx]
        x = torch.cat([ctx, masks], dim=1)
        x = self.blocks(x)
        return self.proj(x[:, n_ctx:])                               # predictions at target slots


class JEPA(nn.Module):
    def __init__(self, encoder, predictor, ema_decay=0.998, stop_grad=True):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)                                   # stop-grad on target
        self.predictor = predictor
        self.ema_decay = ema_decay
        self.stop_grad = stop_grad                                    # the load-bearing switch

    def forward(self, x, context_idx, target_idx):
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

    def step_ema(self):
        ema_update(self.target_encoder, self.context_encoder, self.ema_decay)


# --------------------------------------------------------------------------- toy data + train
def make_batch(B, img=16, device="cpu"):
    """Spatially-correlated images (smoothed noise) so context is predictive of targets."""
    x = torch.randn(B, 1, img, img, device=device)
    x = F.avg_pool2d(x, 5, stride=1, padding=2)                       # blur -> local structure
    return (x - x.mean()) / (x.std() + 1e-6)


def train(decay, stop_grad=True, steps=300, B=128, img=16, patch=4, d=64, seed=0, device="cpu"):
    torch.manual_seed(seed)
    enc = TinyEncoder(img, patch, d)
    grid = img // patch
    pred = TinyPredictor(grid * grid, d)
    model = JEPA(enc, pred, ema_decay=decay, stop_grad=stop_grad).to(device)
    opt = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=1e-3)

    traj = []
    for step in range(steps):
        x = make_batch(B, img, device)
        ctx_idx, tgt_idx = random_block_mask(grid, block=2, device=device)
        loss, tgt = model(x, ctx_idx, tgt_idx)
        opt.zero_grad(); loss.backward(); opt.step()
        model.step_ema()
        if step % 50 == 0 or step == steps - 1:
            std = tgt.detach().std(dim=0).mean().item()               # collapse detector
            traj.append((step, loss.item(), std))
    return traj


if __name__ == "__main__":
    print("Day-2 build: WHAT actually prevents collapse? (corrected after the first run)\n")
    print("Hypothesis under test: it's the STOP-GRADIENT, not the EMA decay (SimSiam, 2020).\n")
    configs = [
        ("no stop-grad (symmetric)", dict(decay=0.998, stop_grad=False)),
        ("stop-grad, decay=0.0",     dict(decay=0.0,   stop_grad=True)),
        ("stop-grad, decay=0.998",   dict(decay=0.998, stop_grad=True)),
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
