"""Phase 1 - scientific-field data module: CAMELS 2D maps -> curated (1,256,256) tensors.

Tasks:
  - curate OFFLINE into a manifest, not inside __getitem__ (deterministic epochs, no
    recursive resampling)
  - threshold read from the DISTRIBUTION'S LOWER TAIL.
  - reject degenerate samples (a near-constant field map == a locked-off video clip:
    trivial prediction, no signal)

MULTI-FIELD: one FieldMapDataset = one field file (its own transform + per-field stats +
manifest). build_multifield() pools 13 fields into ONE ~195k-map SSL corpus via
ConcatDataset -> per-field standardization is automatic (composition), one 2D ViT.

PARAM INFERENCE (probe): each map carries 6 params = 2 cosmological (Omega_m, sigma_8)
+ 4 astrophysical feedback. Pretraining is unsupervised (params off); the probe turns
params on. Baseline to beat = CAMELS `o3_err` supervised CNN (few-% accuracy).

Data: CAMELS Multifield Dataset (CMD). One field file `Maps_<field>_<suite>_<set>_z=0.00.npy`
is an (N, 256, 256) float32 array (N=15000 for the LH set). Values span orders of magnitude
-> a log-transform is the field-specific curation decision (cf. motion-normalize for video).
Docs: https://camels-multifield-dataset.readthedocs.io/en/latest/data.html
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

EPS = 1e-6  # single log-floor constant, shared by _transform and analyze_fields


class FieldMapDataset(Dataset):
    def __init__(self, npy_path: str, name: str = "field", transform: str = "log10",
                 manifest=None, min_std=None, size: int = 256,
                 params_path: str = None, return_params: bool = False,
                 use_cache: bool = True):
        """
        Initializes the dataset object, establishes the memory-mapped link to the physical file,
        and orchestrates the dataset's preparation phase. It automatically triggers the offline
        curation process (if no manifest is provided) and computes the global mean and standard
        deviation required for normalizing the data before training.
        """
        # mmap so 3.9GB doesn't land in RAM; shape (N, 256, 256)
        self.npy_path = npy_path
        self.maps = np.load(npy_path, mmap_mode="r")
        self.name = name                      # field name (e.g. "Mgas", "Vgas") - for stats/probe
        self.transform = transform            # "log10" | "asinh" | "none" (see _transform)
        self.size = size
        self.min_std = min_std
        self.use_cache = use_cache            # cache the (manifest, mean, std) to disk, skip re-scan

        # Optional labels for the probe: (n_sims, 6) params, ONE ROW PER SIM.
        self.return_params = return_params
        self.params = self._load_params(params_path) if params_path else None

        # map -> sim mapping. N maps are split evenly across n_sims (15/sim for CAMELS LH).
        # Derive the divisor so a different set/suite self-checks, instead of hardcoding 15.
        # (Map ORDERING within a sim is still a VERIFY item vs CMD data.py.)
        if self.params is not None:
            self.maps_per_sim = len(self.maps) // len(self.params)
        else:
            self.maps_per_sim = 15

        # ONE offline pass: curate (build the manifest) AND accumulate the standardization
        # mean/std over the surviving maps. Populates self.manifest / self.mean / self.std.
        self.manifest = None
        self.mean = None
        self.std = None
        self._prepare(manifest)

    # --------------------------------------------------------------------- curation (offline)
    def _transform(self, m: np.ndarray) -> np.ndarray:
        """
        Applies a specified mathematical transformation ('log10', 'asinh', or 'none') to raw 
        scientific data to normalize variations that span multiple orders of magnitude. It ensures 
        that physically positive fields (like gas mass) receive a log transform with a safety floor, 
        while signed fields (like velocities) receive an inverse hyperbolic sine transform to 
        prevent NaN errors.
        """              
        if self.transform == "log10":
            m = np.clip(m, a_min=EPS, a_max=None)
            return np.log10(m)
        
        elif self.transform == "asinh":
            ASINH_SCALE = 1e-6
            return np.arcsinh(m / ASINH_SCALE)
        
        elif self.transform == "none":
            return m

        raise ValueError(f"[{self.name}] unknown transform '{self.transform}' "
                         f"(expected 'log10' | 'asinh' | 'none').")

    def _load_map(self, i: int) -> np.ndarray:
        """
        Reads a single 2D spatial map from the memory-mapped array, converts it to a standard 
        float32 format, and applies the designated transformation. This serves as the primary 
        data-fetching mechanism, keeping memory usage strictly limited to one map at a time.
        """
        m = self.maps[i].astype(np.float32)
        return self._transform(m)

    def _load_params(self, params_path: str) -> np.ndarray:
        """
        Parses and loads the cosmological and astrophysical simulation parameters from a text 
        file into a NumPy array. It stores the label data needed for downstream parameter 
        inference (the probe task) so it can be mapped to individual field slices.
        """
        return np.loadtxt(params_path, dtype=np.float32)

    def _curate(self, m: np.ndarray) -> bool:
        """
        Evaluates a single map to determine if it contains enough spatial variation (signal) 
        to be useful for the model. Returns True if the map's standard deviation meets or 
        exceeds the min_std threshold, and False if it is a degenerate, uniform "dead" map.
        """
        if self.min_std is None:
            return True
        return np.std(m) >= self.min_std

    # --------------------------------------------------------------------- manifest cache (disk)
    def _cache_path(self) -> str:
        "Cache sits next to the .npy (on the persistent volume) so it survives pod restarts."
        return self.npy_path + ".manifest.npz"

    def _min_std_key(self) -> float:
        "None is not npz-storable; encode 'no threshold' as -1.0 for the validity check."
        return -1.0 if self.min_std is None else float(self.min_std)

    def _load_cache(self) -> bool:
        """Load a previously saved (manifest, mean, std) if present and still valid.

        Invalidates automatically if the .npy is newer than the cache, or if the curation
        config that produced it (transform / min_std / map count / map shape) no longer matches.
        Returns True on a hit (self.manifest/mean/std populated), False otherwise.
        """
        if not self.use_cache:
            return False
        path = self._cache_path()
        if not os.path.exists(path):
            return False
        if os.path.getmtime(path) < os.path.getmtime(self.npy_path):   # stale: data changed
            return False
        try:
            z = np.load(path, allow_pickle=False)
            H, W = self.maps.shape[1], self.maps.shape[2]
            if (int(z["n_maps"]) != len(self.maps)
                    or str(z["transform"]) != self.transform
                    or float(z["min_std"]) != self._min_std_key()
                    or int(z["H"]) != H or int(z["W"]) != W):
                return False
            self.manifest = z["manifest"].tolist()
            self.mean = float(z["mean"])
            self.std = float(z["std"])
        except Exception:
            return False   # any corruption -> fall back to a fresh scan
        print(f"[{self.name}] loaded cached manifest ({len(self.manifest)} maps, "
              f"mean={self.mean:.4f}, std={self.std:.4f}).")
        return True

    def _save_cache(self):
        "Atomic write (tmp -> os.replace) so a concurrent rank never reads a half-written file."
        if not self.use_cache:
            return
        H, W = self.maps.shape[1], self.maps.shape[2]
        path = self._cache_path()
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            np.savez(tmp, manifest=np.asarray(self.manifest, dtype=np.int64),
                     mean=self.mean, std=self.std, n_maps=len(self.maps),
                     transform=self.transform, min_std=self._min_std_key(), H=H, W=W)
            os.replace(tmp, path)
        except OSError:
            # read-only volume or a lost write race -> skip caching, never crash training
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _prepare(self, manifest):
        """Single offline pass over the field file: curate to build the manifest AND
        accumulate the standardization mean/std over the SURVIVING maps at the same time.

        Replaces the old 3-pass path (build manifest, then 2-pass mean/std) with 1 pass.
        Variance via E[x^2] - E[x]^2 with float64 accumulators; the data is log10/asinh-
        scaled to ~O(10), so catastrophic cancellation is not a concern at this range.

        manifest is None  -> curate every map to build it.
        manifest supplied -> trust it, just accumulate stats over those indices.

        H, W are read from the array shape, not the `size` arg -> a non-256^2 field can't
        silently corrupt the pixel count (and thus the stats).
        """
        build = manifest is None
        if build and self._load_cache():        # reruns: load in ms instead of re-scanning ~4GB
            return
        H, W = self.maps.shape[1], self.maps.shape[2]
        indices = range(len(self.maps)) if build else manifest

        kept = []
        total_sum = 0.0
        total_sumsq = 0.0
        for i in indices:
            m = self._load_map(i)
            if build and not self._curate(m):
                continue
            kept.append(i)
            m64 = m.astype(np.float64, copy=False)
            total_sum += float(m64.sum())
            total_sumsq += float((m64 * m64).sum())

        if not kept:
            raise ValueError(f"[{self.name}] manifest is empty. Adjust min_std or check data.")

        self.manifest = kept
        total_pixels = len(kept) * H * W
        mean = total_sum / total_pixels
        var = total_sumsq / total_pixels - mean * mean

        self.mean = float(mean)
        # var can go slightly negative from rounding on a near-flat field -> clamp.
        self.std = float(np.sqrt(var)) if var > 1e-16 else 1.0

        if build:
            print(f"[{self.name}] curation: retained {len(kept)} / {len(self.maps)} maps "
                  f"(mean={self.mean:.4f}, std={self.std:.4f}).")
            self._save_cache()               # persist so the next run skips this scan entirely

    # --------------------------------------------------------------------- torch Dataset API
    def __len__(self):
        """
        Returns the total number of valid, curated maps available for training. Required by 
        the PyTorch Dataset API so the DataLoader knows the boundary of the dataset.
        """
        return len(self.manifest)

    def __getitem__(self, i):
        """
        Retrieves, standardizes, and formats a single curated map (and optionally its simulation 
        parameters) for model ingestion. Maps the sequential dataloader index to the true 
        manifest index, converts the array to a PyTorch tensor, and adds the channel dimension.
        """
        real_idx = self.manifest[i]

        m = self._load_map(real_idx)
        m = (m - self.mean) / self.std
        x = torch.from_numpy(m).float().unsqueeze(0)         # (1, H, W)

        if self.return_params:
            sim_idx = real_idx // self.maps_per_sim
            y = torch.from_numpy(self.params[sim_idx]).float()
            return x, y
        return x


def build_multifield(field_configs, batch_size: int = 64, num_workers: int = 8,
                     return_params: bool = False, **ds_kw):
    """
    Aggregates multiple distinct physical fields (e.g., Gas Mass, Gas Velocity) into a single, 
    unified dataset corpus. Iterates over a list of field configurations, creating an individual 
    FieldMapDataset for each, stitches them together using PyTorch's ConcatDataset, and wraps 
    them in a DataLoader.
    """
    datasets = []

    for config in field_configs:
        kwargs = {**ds_kw, **config, "return_params": return_params}
        datasets.append(FieldMapDataset(**kwargs))

    # ConcatDataset inherently handles routing __getitem__ to the correct underlying dataset.
    pooled_dataset = ConcatDataset(datasets)

    return DataLoader(pooled_dataset, batch_size = batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)


def analyze_fields(npy_path: str, n: int = 2000, transform: str = "log10"):
    """
    Acts as a diagnostic and curation-decision helper tool by analyzing the statistical 
    distribution of spatial variations within a field. It samples the dataset, prints standard 
    deviation percentiles, estimates map rejection rates, and flags negative values to help 
    select the correct transform and min_std threshold.
    """
    maps = np.load(npy_path, mmap_mode="r")
    total_maps = len(maps)

    # Randomly sample 'n' fields without replacement
    sample_size = min(n, total_maps)
    indices = np.random.choice(total_maps, size=sample_size, replace=False)

    stds = []
    global_raw_min = float('inf')

    for i in indices:
        m_raw = maps[i].astype(np.float32)
        global_raw_min = min(global_raw_min, np.min(m_raw))

        if transform == "log10":
            m = np.clip(m_raw, a_min=EPS, a_max=None)
            m = np.log10(m)
        elif transform == "asinh":
            m = np.arcsinh(m_raw / EPS)
        else:
            m = m_raw

        stds.append(np.std(m))

    stds = np.array(stds)
    percentiles = [5, 10, 25, 50, 75, 95]
    p_vals = np.percentile(stds, percentiles)

    # Output the distribution for visual inspection
    print(f"--- Field Distribution Analysis ({sample_size} maps sampled) ---")
    print(f"Raw Minimum Value: {global_raw_min:.6f}")
    if global_raw_min < 0 and transform == "log10":
        print(">> WARNING: Negative values detected! The 'log10' transform will produce NaNs. Switch to 'asinh'. <<\n")

    print("Standard Deviation Percentiles:")
    for p, val in zip(percentiles, p_vals):
        tail_label = " <-- Candidate Reject Threshold" if p in [5, 10] else ""
        print(f"  p{p:<2}: {val:.6f}{tail_label}")

    print("\nImpact Estimation:")
    for p, val in zip([5, 10], p_vals[:2]):
        drop_rate = np.mean(stds < val) * 100
        print(f"  If min_std = {val:.6f} (p{p}), you will reject ~{drop_rate:.1f}% of maps.")


def build_loader(npy_path: str, batch_size: int = 64, num_workers: int = 8, **kw):
    ds = FieldMapDataset(npy_path, **kw)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, drop_last=True)


if __name__ == "__main__":
    path = r"C:\Users\Koushik\vjepa2-probe\samples\Maps_B_IllustrisTNG_LH_z=0.00.npy"
    analyze_fields(path)
