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

from jepa_loss import TinyEncoder

# --------------------------------------------------------------------- frozen encoder
def load_frozen_encoder(ckpt_path: str, device, **enc_kw) -> nn.Module:
    """
    Instantiates the ViT encoder and loads pretrained FSDP/DDP weights.

    Behavior:
        Initializes the `TinyEncoder` configuration, loads the state dict, and strips 
        away any prefix strings like `module.`, `_fsdp_wrapped.`, or `context_encoder.` 
        that were injected by the distributed training wrappers. Sets to eval mode 
        and freezes all parameters.
        
    Role in Program:
        Provides the fixed representation extractor. The JEPA encoder has finished 
        learning; this function prepares it to embed data for the downstream probe.
    """
    encoder = TinyEncoder(**enc_kw).to(device)
    
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        clean_dict = {}
        for k, v in state_dict.items():
            # Strip distributed / JEPA-wrapper prefixes
            clean_k = k.replace("module.", "").replace("_fsdp_wrapped.", "").replace("context_encoder.", "")
            clean_dict[clean_k] = v
        encoder.load_state_dict(clean_dict, strict=False)
        
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
        
    return encoder


# --------------------------------------------------------------------- probe head
class ProbeHead(nn.Module):
    """
    A lightweight Multi-Layer Perceptron predicting posterior moments.

    Behavior:
        Takes a sequence of patch tokens, applies mean-pooling to collapse them into a 
        single global vector, and passes this through a hidden layer. It outputs two 
        values per predicted parameter: a predicted mean (mu) and a strictly positive 
        standard deviation (sigma, enforced via softplus).
        
    Role in Program:
        The only trainable part of Phase 1 EVAL. It maps the unsupervised JEPA 
        representations into explicit cosmological parameter predictions and their 
        associated uncertainties.
    """

    def __init__(self, d: int, n_params: int = 2):
        super().__init__()
        self.n_params = n_params
        
        # Simple MLP structure
        self.mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 2 * n_params)
        )

    def forward(self, tokens: torch.Tensor):
        # Pool the sequence of tokens: (B, n_tokens, d) -> (B, d)
        pooled = tokens.mean(dim=1)
        
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
    
    # Aggregate with log to stabilize and match CMD benchmark behavior
    loss = torch.log(term1).mean() + torch.log(term2).mean()
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
    """

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
def train_probe(encoder, head, train_loader, val_loader, device, epochs: int = 20):
    """
    Executes the training loop for the parameter-inference probe head.

    Behavior:
        Iterates over the dataset, freezing the encoder and passing its embeddings 
        to the probe head. Standardizes the ground truth targets (first 2 columns: 
        Omega_m and sigma_8) to zero mean and unit variance so the loss function 
        treats them equally. Optimizes the moment loss.
        
    Role in Program:
        Teaches the small MLP how to read the JEPA latent space to extract cosmology.
    """

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Ideally, extract these stats from the train dataset before loop
    # Hardcoded dummy stats for standardizing Omega_m and sigma_8 here:
    target_mean = torch.tensor([0.3, 0.8], device=device) 
    target_std = torch.tensor([0.1, 0.1], device=device)
    
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            # Take only col 0 (Omega_m) and col 1 (sigma_8)
            y = batch_y[:, :2].to(device)
            y_norm = (y - target_mean) / target_std
            
            with torch.no_grad():
                tokens = encoder(batch_x)
                
            mu, sigma = head(tokens)
            loss = moment_loss(mu, sigma, y_norm)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f}")


@torch.no_grad()
def eval_probe(encoder, head, loader, device):
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
    
    target_mean = torch.tensor([0.3, 0.8], device=device) 
    target_std = torch.tensor([0.1, 0.1], device=device)
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        y = batch_y[:, :2].to(device)
        
        tokens = encoder(batch_x)
        mu_norm, sigma_norm = head(tokens)
        
        # De-standardize back to physical units
        mu = (mu_norm * target_std) + target_mean
        sigma = sigma_norm * target_std
        
        all_mu.append(mu)
        all_sigma.append(sigma)
        all_y.append(y)
        
    mu = torch.cat(all_mu, dim=0)
    sigma = torch.cat(all_sigma, dim=0)
    y = torch.cat(all_y, dim=0)
    
    # Calculate Metrics
    rmse = torch.sqrt(torch.mean((y - mu)**2, dim=0))
    ss_tot = torch.sum((y - torch.mean(y, dim=0))**2, dim=0)
    ss_res = torch.sum((y - mu)**2, dim=0)
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
        global_omega.append(batch_y[:, 0].numpy()) # Omega_m
        
        # Save first map in first batch for spatial back-map
        if sample_tokens is None:
            sample_tokens = tokens[0].cpu().numpy() # (n_tokens, d)
            sample_map = batch_x[0, 0].cpu().numpy()

    # (a) GLOBAL atlas
    X = np.concatenate(global_embeds, axis=0)
    C = np.concatenate(global_omega, axis=0)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
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
    #   enc  = load_frozen_encoder("ckpt.pt", device)
    #   head = ProbeHead(d=enc_dim).to(device)
    #   tr, va, te = sim_split(len(dataset))
    #   train_probe(enc, head, tr_loader, va_loader, device)
    #   print(eval_probe(enc, head, te_loader_ITNG, device))   # in-suite
    #   print(eval_probe(enc, head, te_loader_SIMBA, device))  # held-out = robustness
    #   latent_atlas(enc, atlas_loader, device, out_dir="notes/atlas")
    pass
