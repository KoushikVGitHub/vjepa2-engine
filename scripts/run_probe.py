"""Driver: train + evaluate the cosmology probe on the frozen ViT-L JEPA encoder.

Runs single-GPU. Rebuilds the encoder at the KEEPER's ViT-L config (d1024 / L24 / heads16,
img256 / patch16) so the saved context_encoder state_dict loads cleanly; load_frozen_encoder
RAISES if any encoder param is missing, which guards against a silently-random encoder from a
config mismatch. Split is at SIMULATION granularity (probe.sim_split) -- never by map.

Usage:
  python scripts/run_probe.py --ckpt /workspace/ckpt.pt --field Mgas --epochs 20
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import Subset, DataLoader, TensorDataset

from data.fields import FieldMapDataset
from probe import (load_frozen_encoder, ProbeHead, sim_split,
                   train_probe, eval_probe, latent_atlas)

# Default ViT-L encoder config -- MUST match the trained run or the state_dict won't load.
# Overridable from the CLI (--patch / --enc-d / --enc-layers / --enc-heads / --img) so a
# patch-8 or downscaled-backbone checkpoint can be probed without editing this file.
ENC = dict(img=256, patch=16, d=1024, heads=16, layers=24)


def make_dataset(data_root, field, suite):
    npy = os.path.join(data_root, f"Maps_{field}_{suite}_LH_z=0.00.npy")
    params = os.path.join(data_root, f"params_LH_{suite}.txt")
    # log10 + min_std=0.05 must match the corpus config so the cached manifest is reused
    # (and, for keep-all fields, manifest is identity -> sim_split map indices == dataset positions).
    return FieldMapDataset(npy, name=field, transform="log10", min_std=0.05,
                           params_path=params, return_params=True, use_cache=True)


def loader(ds, idx, bs, shuffle, workers):
    return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=shuffle,
                      num_workers=workers, drop_last=False, pin_memory=True)


@torch.no_grad()
def precompute_tokens(encoder, ds, idx, device, bs, workers):
    """Run the FROZEN encoder over these maps ONCE and cache its tokens (bf16, on CPU).

    The probe re-encodes the same maps every epoch; since the encoder is frozen that is pure waste.
    Encode once here (bf16 autocast, large batch, no grad) into a TensorDataset of (tokens, y); the
    head then trains on the cache with encoder=None. ~epochs*forwards -> 1 forward = the whole
    speedup. Tokens are (N, n_patch, d): ~6GB bf16 for a 12k-map train split at ViT-L -> kept on CPU,
    moved per batch. Deterministic because the probe dataset has augment OFF, so the cache is exact.
    """
    dl = DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
                    num_workers=workers, drop_last=False, pin_memory=True)
    toks, ys = [], []
    use_cuda = torch.device(device).type == "cuda"
    for x, y in dl:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            t = encoder(x)
        toks.append(t.to(torch.bfloat16).cpu())
        ys.append(y)
    return TensorDataset(torch.cat(toks), torch.cat(ys))


def cached_loader(tds, bs, shuffle):
    # data already in RAM -> workers=0 (no decode), pin for the H2D copy.
    return DataLoader(tds, batch_size=bs, shuffle=shuffle, drop_last=False, pin_memory=True)


def _feat_sig(cache_dir, ckpt, enc_cfg, field, suite, split, probe_stage, random_init, seed):
    """Disk path for a split's cached features. The key covers everything that changes the tokens:
    ckpt identity (abspath + mtime + size), encoder geometry, field/suite/split, and probe-stage.
    It includes --seed ONLY for a random-init encoder -- those weights come from the torch RNG, so
    they are seed-dependent; a *trained* encoder's forward is deterministic AND sim_split has a fixed
    internal seed=0, so its features are seed-independent and head-seed reprobes share one cache."""
    if random_init:
        ck_id = f"randominit:seed{seed}"
    else:
        try:
            st = os.stat(ckpt)
            ck_id = f"{os.path.abspath(ckpt)}:{int(st.st_mtime)}:{st.st_size}"
        except OSError:
            ck_id = os.path.abspath(ckpt)
    geom = "-".join(f"{k}{enc_cfg[k]}" for k in sorted(enc_cfg))
    key = f"{ck_id}|{geom}|{field}|{suite}|{split}|{probe_stage}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"feat_{field}_{suite}_{split}_{probe_stage}_{h}.pt")


@torch.no_grad()
def cached_tokens(encoder, ds, idx, device, bs, workers, path):
    """Disk-persisted wrapper around precompute_tokens: load the split's features if the cache file
    exists, else run the encoder ONCE and save them atomically. Lets repeat probes (head-seeds,
    ridge-alpha sweeps, CPU-only refits) skip the encoder pass entirely. The saved tensors are the
    exact device-computed bf16 tokens, so a later CPU load yields identical features -- no fp32/bf16
    skew, so R^2 stays comparable to a GPU run. path=None disables the disk layer (in-RAM only)."""
    if path and os.path.exists(path):
        obj = torch.load(path, map_location="cpu")
        print(f"[probe] cached features HIT  <- {os.path.basename(path)} {tuple(obj['x'].shape)}")
        return TensorDataset(obj["x"], obj["y"])
    tds = precompute_tokens(encoder, ds, idx, device, bs, workers)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"        # atomic: a killed run leaves no partial cache file
        torch.save({"x": tds.tensors[0], "y": tds.tensors[1]}, tmp)
        os.replace(tmp, path)
        print(f"[probe] cached features SAVED -> {os.path.basename(path)}")
    return tds


def _print_metrics(res):
    for k, v in res.items():
        print(f"  {k:9s}: Omega_m={v[0]:.4f}   sigma8={v[1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt.pt")
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--field", default="Mgas")
    # Encoder geometry -- MUST match the trained checkpoint (esp. --patch: patch-8 vs patch-16
    # gives a different patch-embed shape, so a mismatch makes load_frozen_encoder RAISE).
    ap.add_argument("--img", type=int, default=ENC["img"])
    ap.add_argument("--patch", type=int, default=ENC["patch"])
    ap.add_argument("--enc-d", type=int, default=ENC["d"])
    ap.add_argument("--enc-heads", type=int, default=ENC["heads"])
    ap.add_argument("--enc-layers", type=int, default=ENC["layers"])
    ap.add_argument("--stem", choices=["linear", "conv", "convdisjoint", "mlp"], default="linear",
                    help="tokenizer -- MUST match how the checkpoint was trained (conv/convdisjoint->"
                         "conv_stem.*, mlp->mlp_stem.*, linear->proj.* keys).")
    ap.add_argument("--stem-pad", choices=["circular", "zeros"], default="circular",
                    help="conv-stem padding -- MUST match training (only used when --stem conv).")
    ap.add_argument("--random-init", action="store_true",
                    help="H7 control: DON'T load the checkpoint -- probe a randomly-initialised frozen "
                         "encoder. trained_R2 - random_R2 = information the pretext task actually taught.")
    ap.add_argument("--probe-stage", choices=["encoder", "tokenizer"], default="encoder",
                    help="H3 control: 'tokenizer' probes the RAW tokenizer output (pre-transformer) to "
                         "localise where the conv gain lives; 'encoder' (default) probes encoded tokens.")
    ap.add_argument("--seed", type=int, default=0, help="seed for deterministic probe (head init + shuffle)")
    ap.add_argument("--suite", default="IllustrisTNG", help="in-suite (pretraining) suite")
    ap.add_argument("--heldout", default="SIMBA", help="cross-suite robustness suite (if on disk)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-atlas", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable frozen-feature caching (re-encode every epoch; ~15x slower)")
    ap.add_argument("--precompute-batch", type=int, default=256,
                    help="batch size for the one-time feature precompute pass (frozen fwd, no grad "
                         "-> can be large)")
    ap.add_argument("--feat-cache-dir", default=None,
                    help="dir for DISK-persisted frozen features (default: <data-root>/_probe_feat_cache). "
                         "First probe of a ckpt extracts + saves; later head-seed / ridge-alpha / CPU "
                         "refits load from disk and skip the encoder pass entirely.")
    ap.add_argument("--no-disk-cache", action="store_true",
                    help="keep the in-RAM feature cache but do NOT read/write the disk cache "
                         "(use when volume space is tight; each split cache is ~0.75-6GB).")
    args = ap.parse_args()

    # Seed the probe (head init + train-loader shuffle both use torch's global RNG) so re-probing
    # the same checkpoint is deterministic -- otherwise probe R^2 varies ~+-0.03 run-to-run, which
    # swamps a convergence curve. Fixed seed => the R^2-vs-steps trend is signal, not head-init noise.
    torch.manual_seed(args.seed)

    # Build the encoder config from the CLI so a patch-8 / downscaled checkpoint loads cleanly.
    # NB: use a distinct name (not ENC) -- assigning ENC here would shadow the module-level ENC
    # for all of main(), breaking the argparse `default=ENC[...]` reads above (UnboundLocalError).
    enc_cfg = dict(img=args.img, patch=args.patch, d=args.enc_d, heads=args.enc_heads,
                   layers=args.enc_layers, stem=args.stem, stem_pad=args.stem_pad)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Disk feature cache: on unless --no-cache (caching off entirely) or --no-disk-cache (RAM only).
    cdir = None if (args.no_cache or args.no_disk_cache) else (
        args.feat_cache_dir or os.path.join(args.data_root, "_probe_feat_cache"))

    def _featpath(suite, split):
        if cdir is None:
            return None
        return _feat_sig(cdir, args.ckpt, enc_cfg, args.field, suite, split,
                         args.probe_stage, args.random_init, args.seed)

    enc = load_frozen_encoder(args.ckpt, device, random_init=args.random_init, **enc_cfg)
    enc.probe_stage = args.probe_stage            # H3: read raw tokenizer output when "tokenizer"
    n_enc = sum(p.numel() for p in enc.parameters())
    tag = "RANDOM-INIT" if args.random_init else "trained"
    print(f"[probe] {tag} encoder ({args.probe_stage} stage): {n_enc / 1e6:.1f}M params (ViT-L, {enc_cfg})")

    ds = make_dataset(args.data_root, args.field, args.suite)
    tr_idx, va_idx, te_idx = sim_split(len(ds))
    print(f"[probe] {args.field}/{args.suite}: {len(ds)} maps -> "
          f"{len(tr_idx)} train / {len(va_idx)} val / {len(te_idx)} test (sim-level split)")

    head = ProbeHead(d=enc_cfg["d"]).to(device)

    # enc_head is what train/eval call: None when features are cached (they get precomputed tokens),
    # else the live encoder. The real `enc` is still used for the one-time precompute + the atlas.
    if args.no_cache:
        enc_head = enc
        tr = loader(ds, tr_idx, args.batch, True, args.workers)
        va = loader(ds, va_idx, 128, False, args.workers)
        te = loader(ds, te_idx, 128, False, args.workers)
    else:
        print("[probe] precomputing frozen features (one bf16 pass over each split)..."
              + ("" if cdir is None else f"  [disk cache: {cdir}]"))
        enc_head = None
        tr = cached_loader(cached_tokens(enc, ds, tr_idx, device, args.precompute_batch, args.workers,
                                         _featpath(args.suite, "train")), args.batch, True)
        va = cached_loader(cached_tokens(enc, ds, va_idx, device, args.precompute_batch, args.workers,
                                         _featpath(args.suite, "val")), 256, False)
        te = cached_loader(cached_tokens(enc, ds, te_idx, device, args.precompute_batch, args.workers,
                                         _featpath(args.suite, "test")), 256, False)

    train_probe(enc_head, head, tr, va, device, epochs=args.epochs)

    print(f"\n=== IN-SUITE ({args.suite}) ===")
    _print_metrics(eval_probe(enc_head, head, te, device))

    # Cross-suite robustness (challenge #2) -- only if the held-out suite is on disk.
    held_npy = os.path.join(args.data_root, f"Maps_{args.field}_{args.heldout}_LH_z=0.00.npy")
    if os.path.exists(held_npy):
        hds = make_dataset(args.data_root, args.field, args.heldout)
        h_idx = list(range(len(hds)))
        if args.no_cache:
            hte = loader(hds, h_idx, 128, False, args.workers)
        else:
            hte = cached_loader(cached_tokens(enc, hds, h_idx, device, args.precompute_batch, args.workers,
                                              _featpath(args.heldout, "heldout")), 256, False)
        print(f"\n=== HELD-OUT ({args.heldout}) = cross-suite robustness ===")
        _print_metrics(eval_probe(enc_head, head, hte, device))
    else:
        print(f"\n[probe] held-out suite {args.heldout} not on disk -- skipping cross-suite eval.")

    if not args.no_atlas:
        try:
            # atlas needs the real encoder + raw maps (spatial back-map), so give it a raw loader.
            atlas_loader = te if args.no_cache else loader(ds, te_idx, 128, False, args.workers)
            latent_atlas(enc, atlas_loader, device, out_dir="notes/atlas")
            print("[probe] latent atlas -> notes/atlas/")
        except Exception as e:
            print(f"[probe] atlas skipped ({type(e).__name__}: {e}); "
                  f"pip install matplotlib scikit-learn to enable.")


if __name__ == "__main__":
    main()
