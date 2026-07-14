# CAMELS 2D field statistics (IllustrisTNG LH, z=0.00)

Empirical justification for the curation/preprocessing decisions in
`src/train_fsdp.py::build_dataloader`. Two tables:

1. **`analyze_all.py` distribution scan** (2000 maps sampled per field) — used to pick the
   transform and confirm no degenerate maps.
2. **Standardization stats** (per-field mean/std over the full curated set, `log10`) — computed
   at `FieldMapDataset` construction and used to normalize inputs to ~N(0,1). Recorded here so
   they don't have to be re-derived from a run.

## Decisions
- **All fields positive-definite → uniform `log10`.** Vgas/Vcdm raw-min = +5.1/+5.3 (velocity
  *magnitudes*, not signed); MgFe +0.013. No `asinh` needed for this dataset.
- **No field has a degenerate/near-constant population** — even p5 std is healthy — so curation
  is effectively keep-all. `min_std = 0.05` is a defensive floor that only rejects a pathological
  fully-empty map (Mstar/Z/ne have raw-min 0). Confirmed at runtime: 15000/15000 retained per field.
- **B dropped**: IllustrisTNG-only + floor-dominated (p50 std 0.009, mean ≈ log-floor).
- **12 fields kept** (all exist in both suites → reusable for the SIMBA held-out probe):
  Mgas, Mcdm, Mtot, Mstar, T, P, Z, HI, ne, MgFe, Vgas, Vcdm.

## 1. analyze_all.py — std percentiles (transform used in the scan)
| field | transform | raw-min | p5 | p10 | p25 | p50 | p75 | p95 |
|-------|-----------|---------|----|-----|-----|-----|-----|-----|
| B *(dropped)* | log10 | 0.000 | 0.00187 | 0.00290 | 0.00530 | 0.00926 | 0.01357 | 0.01901 |
| HI | log10 | 14.656 | 0.791 | 0.833 | 0.906 | 0.976 | 1.023 | 1.081 |
| Mcdm | log10 | 2.872e9 | 0.375 | 0.396 | 0.426 | 0.460 | 0.493 | 0.536 |
| MgFe | log10 | 0.0126 | 0.109 | 0.121 | 0.143 | 0.163 | 0.185 | 0.218 |
| Mgas | log10 | 8.542e8 | 0.368 | 0.392 | 0.427 | 0.461 | 0.497 | 0.550 |
| Mstar | log10 | 0.000 | 0.856 | 1.006 | 1.256 | 1.582 | 1.963 | 2.591 |
| Mtot | log10 | 5.886e9 | 0.370 | 0.388 | 0.425 | 0.459 | 0.490 | 0.538 |
| P | log10 | 6.246 | 0.987 | 1.048 | 1.149 | 1.248 | 1.333 | 1.452 |
| T | log10 | 1568.85 | 0.425 | 0.483 | 0.612 | 0.738 | 0.834 | 0.929 |
| Vcdm | asinh† | 5.287 | 0.310 | 0.344 | 0.388 | 0.435 | 0.486 | 0.555 |
| Vgas | asinh† | 5.096 | 0.308 | 0.333 | 0.377 | 0.422 | 0.469 | 0.543 |
| Z | log10 | 0.000 | 0.568 | 0.674 | 0.859 | 1.102 | 1.273 | 1.392 |
| ne | log10 | 0.000 | 0.231 | 0.245 | 0.271 | 0.306 | 0.341 | 0.393 |

† Vgas/Vcdm were scanned under `asinh` (before we confirmed they're positive), so these
percentiles are asinh-space. They are trained under **`log10`** — see the mean/std below,
which are the log10 values actually used.

## 2. Standardization stats used in training (log10, full curated set)
Applied in `FieldMapDataset.__getitem__` as `(x - mean) / std`. Deterministic given the data.

| field | mean | std |
|-------|------|-----|
| Mgas | 10.4148 | 0.4913 |
| Mcdm | 10.9840 | 0.5081 |
| Mtot | 11.1038 | 0.4916 |
| Mstar | -5.8053 | 1.7452 |
| T | 4.2234 | 0.8163 |
| P | 3.6166 | 1.3304 |
| Z | -5.1556 | 1.1942 |
| HI | 4.7847 | 1.0113 |
| ne | -5.8237 | 0.3180 |
| MgFe | 0.0103 | 0.1744 |
| Vgas | 2.0851 | 0.2469 |
| Vcdm | 2.0976 | 0.2550 |

Notes:
- Mstar/Z/ne means near the log-floor (−5 to −6) reflect large empty (zero) regions clipped to
  `EPS`; their high std (Mstar 1.75, Z 1.19) shows they still carry real spatial structure
  (unlike B, which is floor-dominated with std 0.009 → dropped).
- These are recomputed identically on every run, so a probe that builds the same
  `FieldMapDataset` normalizes consistently with pretraining. (A future optimization is to cache
  a manifest+stats file so the per-rank curation pass isn't repeated.)
