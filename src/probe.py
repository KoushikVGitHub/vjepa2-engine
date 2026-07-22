"""Phase 1 EVAL - probe: does the frozen JEPA encoder actually encode cosmology?

Two deliverables (both answer NAMED CAMELS challenges):
  (1) QUANTITATIVE: freeze the pretrained encoder -> small (mu, sigma) moment head ->
      regress the 2 cosmological params (Omega_m, sigma_8). Report IN-SUITE (IllustrisTNG)
      vs HELD-OUT SUITE (SIMBA) = challenge #2 (cross-simulation robustness).
      Baseline to beat = CAMELS `o3_err` supervised CNN (few-% accuracy).
  (2) VISUAL: latent atlas - PCA/UMAP of embeddings colored by Omega_m, plus a spatial
      per-map back-map (Paper 3 style) = challenge #4 (interpretability).

============================ CRITICAL CORRECTNESS NOTE ============================
Split by SIMULATION, never by map. 15 maps share one sim's 6 params (map -> sim = idx//15),
so a map-level train/test split LEAKS the label. Build splits at sim granularity, then expand
each sim to its 15 map indices. Getting this wrong inflates every number.
=================================================================================
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_loss import ViTEncoder

# CAMELS LH prior midpoints for (Omega_m, sigma_8), used to standardize the probe targets so
# the two params are balanced in the loss. Shared by train_probe/eval_probe so the forward
# normalization and the inverse de-standardization can NEVER silently disagree (they used to
# be hardcoded separately in both). Ideally recompute from the train split; these prior-based
# constants are a safe default.
TARGET_MEAN = (0.3, 0.8)
TARGET_STD = (0.1, 0.1)


# --------------------------------------------------------------------- frozen encoder
def load_frozen_encoder(ckpt_path: str, device, **enc_kw) -> nn.Module:
    """
    Instantiates the ViT encoder and loads pretrained FSDP/DDP weights.

    Behavior:
        Initializes the `ViTEncoder` configuration, extracts ONLY the context-encoder
        weights from the checkpoint (discarding predictor / target / wrapper prefixes),
        and loads them. Sets to eval mode and freezes all parameters. Refuses to proceed
        if any encoder parameter was not populated (guards a silently-random encoder).

    Role in Program:
        Provides the fixed representation extractor. The JEPA encoder has finished
        learning; this function prepares it to embed data for the downstream probe.
    """
    encoder = ViTEncoder(**enc_kw).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)  # bundles an args dict
    # Accept either a bare state_dict or a {"model": state_dict, ...} training checkpoint.
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Keep ONLY the context-encoder weights and remap them into ViTEncoder's key space by
    # splitting on the "context_encoder." marker -- this discards ANY wrapper prefix in front
    # of it (module. / _fsdp_wrapped_module. / jepa. ...), which chained .replace() would miss.
    # (Save a gathered FULL_STATE_DICT from FSDP, not the per-rank sharded state.)
    marker = "context_encoder."
    enc_sd = {k.split(marker, 1)[1]: v for k, v in state_dict.items() if marker in k}
    if not enc_sd:
        raise KeyError(f"no '{marker}' keys in {ckpt_path}; sample keys = {list(state_dict)[:4]}")

    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    # The real danger is a SILENTLY random encoder (strict=False hides a prefix mismatch).
    # A non-empty `missing` means some encoder params were never loaded -> refuse to proceed.
    if missing:
        raise RuntimeError(f"encoder params not loaded (still random!): {missing}")
    if unexpected:
        print(f"[probe] ignored {len(unexpected)} unexpected checkpoint keys")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


# --------------------------------------------------------------------- probe head
class AttentivePool(nn.Module):
    """Cross-attention pooling: a single learnable query attends over the token sequence.

    Mean-pooling weights every patch equally, which dilutes the few tokens that actually carry
    the cosmological signal (and is especially lossy on a low-rank representation). A learnable
    query with multi-head attention lets the probe LEARN which tokens to read -- the DINOv2 /
    I-JEPA linear-eval standard. Still cheap and, crucially, the encoder stays frozen: only this
    pool + the MLP train, so it remains an honest test of the frozen features.
    """

    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(tokens.size(0), -1, -1)      # (B, 1, d)
        pooled, _ = self.attn(q, tokens, tokens)           # (B, 1, d)
        return self.norm(pooled.squeeze(1))                # (B, d)


class ProbeHead(nn.Module):
    """
    A lightweight pooling + MLP head predicting posterior moments.

    Behavior:
        Pools a sequence of patch tokens into one global vector -- via attentive pooling
        (default) or mean-pooling -- and passes it through a hidden layer. It outputs two
        values per predicted parameter: a predicted mean (mu) and a strictly positive
        standard deviation (sigma, enforced via softplus).

    Role in Program:
        The only trainable part of Phase 1 EVAL. It maps the unsupervised JEPA
        representations into explicit cosmological parameter predictions and their
        associated uncertainties.
    """

    def __init__(self, d: int, n_params: int = 2, pool: str = "attn", heads: int = 8):
        super().__init__()
        self.n_params = n_params
        self.pool_mode = pool
        self.pool = AttentivePool(d, heads=heads) if pool == "attn" else None

        # Simple MLP structure
        self.mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 2 * n_params)
        )

    def forward(self, tokens: torch.Tensor):
        # Pool the sequence of tokens: (B, n_tokens, d) -> (B, d)
        pooled = self.pool(tokens) if self.pool is not None else tokens.mean(dim=1)

        # Output is (B, 2 * n_params)
        out = self.mlp(pooled)

        mu = out[:, :self.n_params]
        # Sigma must be strictly positive. Add a small epsilon to avoid log(0) down the line.
        sigma = F.softplus(out[:, self.n_params:]) + 1e-6

        return mu, sigma


# --------------------------------------------------------------------- moment loss
def moment_loss(mu, sigma, y):
    """
    Calculates the distribution-free posterior moment loss (Villaescusa-Navarro et al.).

    Behavior:
        Computes two terms without assuming a Gaussian likelihood constraint. Term 1
        pulls `mu` towards the ground truth `y`. Term 2 pulls `sigma^2` towards the
        residual squared error. Taking the logarithm scales it properly.

    Role in Program:
        Trains the probe head to output both highly accurate point-estimates (mu) and
        reliable, calibrated uncertainty boundaries (sigma).
    """
    # term1: pull mu to posterior mean
    term1 = torch.mean((y - mu) ** 2, dim=0)

    # term2: pull sigma^2 to posterior variance
    term2 = torch.mean(((y - mu) ** 2 - sigma ** 2) ** 2, dim=0)

    # Aggregate with log to stabilize and match CMD benchmark behavior (eps guards log(0)).
    loss = torch.log(term1 + 1e-8).mean() + torch.log(term2 + 1e-8).mean()
    return loss


# --------------------------------------------------------------------- sim-level split
def sim_split(n_maps: int, maps_per_sim: int = 15, fracs=(0.8, 0.1, 0.1), seed: int = 0):
    """
    Partitions the dataset into train/val/test splits strictly at the simulation level.

    Behavior:
        Calculates the total number of physical simulations, shuffles their IDs, and
        splits them according to the provided fractions. It then mathematically expands
        those simulation IDs back out into their corresponding map indices.

    Role in Program:
        CRITICAL correctness gate. Since 15 maps are generated from the exact same
        cosmological parameters, a naive random split would place nearly identical
        targets in both train and test sets (data leakage). This prevents that.

    NOTE: single-field only. This assumes contiguous maps where map idx // maps_per_sim = sim.
    For a pooled multi-field ConcatDataset that mapping breaks across field boundaries, so run
    the probe per field (or split each field, then offset its indices into the concat).
    """
    assert n_maps % maps_per_sim == 0, (
        f"n_maps={n_maps} is not a multiple of maps_per_sim={maps_per_sim}; sim_split expects a "
        f"SINGLE field's contiguous maps (idx//15 = sim). For a pooled ConcatDataset, split per field.")

    np.random.seed(seed)
    n_sims = n_maps // maps_per_sim
    sim_indices = np.random.permutation(n_sims)

    n_train = int(fracs[0] * n_sims)
    n_val = int(fracs[1] * n_sims)

    train_sims = sim_indices[:n_train]
    val_sims = sim_indices[n_train:n_train + n_val]
    test_sims = sim_indices[n_train + n_val:]

    def expand(sim_list):
        # [s * 15, s * 15 + 1, ..., s * 15 + 14]
        expanded = [s * maps_per_sim + i for s in sim_list for i in range(maps_per_sim)]
        return expanded

    return expand(train_sims), expand(val_sims), expand(test_sims)


# --------------------------------------------------------------------- train / eval
def _embed(encoder, batch_x, device):
    """Tokens for a batch. encoder=None => batch_x is ALREADY the precomputed tokens.

    The encoder is frozen, so its output is identical every epoch -> compute it ONCE (see
    precompute_tokens in scripts/run_probe.py) and train the head on the cache. That turns ~epochs*
    full ViT-L forwards into a single pass -- the dominant probe cost. When encoder is not None we
    fall back to the live (no-grad) forward, so the un-cached path still works unchanged.
    """
    if encoder is None:
        return batch_x.to(device).float()                 # cached tokens (stored bf16) -> fp32 head
    with torch.no_grad():
        return encoder(batch_x.to(device))


def train_probe(encoder, head, train_loader, val_loader, device, epochs: int = 20,
                target_mean=TARGET_MEAN, target_std=TARGET_STD):
    """
    Executes the training loop for the parameter-inference probe head.

    Behavior:
        Iterates over the dataset, freezing the encoder and passing its embeddings
        to the probe head. Standardizes the ground truth targets (first 2 columns:
        Omega_m and sigma_8) to zero mean and unit variance so the loss function
        treats them equally. Optimizes the moment loss, reporting a validation loss
        each epoch when a val_loader is supplied.

    Role in Program:
        Teaches the small MLP how to read the JEPA latent space to extract cosmology.
    """
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    tm = torch.tensor(target_mean, device=device)
    ts = torch.tensor(target_std, device=device)

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            y = batch_y[:, :2].to(device)          # col 0 = Omega_m, col 1 = sigma_8
            y_norm = (y - tm) / ts

            tokens = _embed(encoder, batch_x, device)
            mu, sigma = head(tokens)
            loss = moment_loss(mu, sigma, y_norm)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        msg = f"Epoch {epoch + 1}/{epochs} | train loss {total_loss / len(train_loader):.4f}"

        # Validation pass (normalized moment loss) so the head isn't flying blind on train only.
        if val_loader is not None:
            head.eval()
            with torch.no_grad():
                vloss = 0.0
                for vx, vy in val_loader:
                    vy_norm = (vy[:, :2].to(device) - tm) / ts
                    vmu, vsig = head(_embed(encoder, vx, device))
                    vloss += moment_loss(vmu, vsig, vy_norm).item()
            msg += f" | val loss {vloss / len(val_loader):.4f}"
        print(msg)


@torch.no_grad()
def eval_probe(encoder, head, loader, device, target_mean=TARGET_MEAN, target_std=TARGET_STD):
    """
    Evaluates the probe head performance and uncertainty calibration.

    Behavior:
        Generates predictions across the provided loader. De-standardizes predictions
        back to raw units. Computes Root Mean Squared Error (RMSE), the coefficient
        of determination (R^2), and coverage probability.

    Role in Program:
        Quantifies Challenge #2 (cross-simulation robustness). Passing a SIMBA loader
        here instantly yields the held-out generalization metrics.
    """
    head.eval()
    all_mu, all_sigma, all_y = [], [], []

    tm = torch.tensor(target_mean, device=device)
    ts = torch.tensor(target_std, device=device)

    for batch_x, batch_y in loader:
        y = batch_y[:, :2].to(device)

        tokens = _embed(encoder, batch_x, device)
        mu_norm, sigma_norm = head(tokens)

        # De-standardize back to physical units
        mu = (mu_norm * ts) + tm
        sigma = sigma_norm * ts

        all_mu.append(mu)
        all_sigma.append(sigma)
        all_y.append(y)

    mu = torch.cat(all_mu, dim=0)
    sigma = torch.cat(all_sigma, dim=0)
    y = torch.cat(all_y, dim=0)

    # Calculate Metrics
    rmse = torch.sqrt(torch.mean((y - mu) ** 2, dim=0))
    ss_tot = torch.sum((y - torch.mean(y, dim=0)) ** 2, dim=0)
    ss_res = torch.sum((y - mu) ** 2, dim=0)
    r2 = 1 - (ss_res / ss_tot)

    # Coverage: Fraction of true labels falling within 1 standard deviation of predicted mean
    coverage = torch.mean((torch.abs(y - mu) <= sigma).float(), dim=0)

    return {
        "RMSE": rmse.cpu().numpy(),
        "R2": r2.cpu().numpy(),
        "Coverage": coverage.cpu().numpy()
    }


# --------------------------------------------------------------------- latent atlas (visual)
@torch.no_grad()
def latent_atlas(encoder, loader, device, out_dir: str):
    """
    Produces global PCA visual maps and spatial feature back-maps.

    Behavior:
        Extracts pooled embeddings and projects them into 2D via PCA (colored by Omega_m).
        For spatial interpretability, it takes the un-pooled token grid of individual maps
        and projects the high-dimensional channels down to 3 (RGB), rendering how the
        model physically "sees" structures.

    Role in Program:
        Solves Challenge #4 (interpretability). Creates visual proof that the JEPA is
        spatially aware and naturally separates universes by their physical parameters.
    """
    import matplotlib.pyplot as plt
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("Scikit-learn required for PCA atlas.")
        return

    os.makedirs(out_dir, exist_ok=True)
    encoder.eval()

    global_embeds, global_omega = [], []
    sample_tokens, sample_map = None, None

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        tokens = encoder(batch_x)
        pooled = tokens.mean(dim=1).cpu().numpy()

        global_embeds.append(pooled)
        global_omega.append(batch_y[:, 0].numpy())  # Omega_m

        # Save first map in first batch for spatial back-map
        if sample_tokens is None:
            sample_tokens = tokens[0].cpu().numpy()  # (n_tokens, d)
            sample_map = batch_x[0, 0].cpu().numpy()

    # (a) GLOBAL atlas
    X = np.concatenate(global_embeds, axis=0)
    C = np.concatenate(global_omega, axis=0)

    # Fit a few components so we can report how concentrated the feature space is: if PC1+PC2
    # explain ~all the variance, the probe's features are effectively rank-2 (independently
    # confirms/refutes the training-time eff_rank collapse signal on the FULL unmasked features).
    n_pc = min(10, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_pc)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    print("[atlas] PCA explained-variance ratio (top {}): {}  | cumulative top-2 = {:.3f}".format(
        min(5, n_pc), ", ".join(f"{r:.3f}" for r in evr[:5]), float(evr[:2].sum())))

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=C, cmap='viridis', alpha=0.6, s=15)
    plt.colorbar(sc, label="Omega_m")
    plt.title("Latent Atlas: PCA of Map Embeddings")
    plt.savefig(os.path.join(out_dir, "global_atlas_pca.png"))
    plt.close()

    # (b) SPATIAL back-map
    # Reshape tokens back to grid (e.g. 16x16 or 64x64 depending on patch size)
    grid_size = int(np.sqrt(sample_tokens.shape[0]))
    tokens_spatial = sample_tokens.reshape(grid_size, grid_size, -1)

    # PCA down to 3 components for RGB
    flat_tokens = tokens_spatial.reshape(-1, tokens_spatial.shape[-1])
    rgb_pca = PCA(n_components=3).fit_transform(flat_tokens)

    # Normalize components to [0, 1] range for image rendering
    rgb_pca = (rgb_pca - rgb_pca.min(axis=0)) / (rgb_pca.max(axis=0) - rgb_pca.min(axis=0))
    rgb_img = rgb_pca.reshape(grid_size, grid_size, 3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(sample_map, cmap='magma')
    axes[0].set_title("Original Input Map")
    axes[1].imshow(rgb_img)
    axes[1].set_title("Spatial Token RGB Back-Map")
    plt.savefig(os.path.join(out_dir, "spatial_backmap.png"))
    plt.close()


if __name__ == "__main__":
    # Wiring sketch (fill once a pretrained checkpoint exists):
    #   enc  = load_frozen_encoder("ckpt.pt", device, img=256, patch=16, d=768, heads=12, layers=12)
    #   head = ProbeHead(d=enc_dim).to(device)
    #   tr, va, te = sim_split(len(dataset))          # single-field dataset
    #   train_probe(enc, head, tr_loader, va_loader, device)
    #   print(eval_probe(enc, head, te_loader_ITNG, device))   # in-suite
    #   print(eval_probe(enc, head, te_loader_SIMBA, device))  # held-out = robustness
    #   latent_atlas(enc, atlas_loader, device, out_dir="notes/atlas")
    pass
