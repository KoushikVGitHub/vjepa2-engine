# Resolving dimensional collapse in LeJEPA on CAMELS

How a run with every "healthy" indicator green was quietly collapsed, why the anti-collapse
regularizer it was trained with couldn't see it, and what fixed it. Ends with a 2.2× probe lift.

---

## 1. The symptom that looked like health

The first ViT-L keeper (1000 steps, λ=0.7, lr 5e-5, FSDP+bf16) reported:

| metric | value | reading at the time |
|---|---|---|
| `tgt_std` | ~0.90, rock-steady | healthy — no complete collapse |
| `reg` (SIGReg) | ~0.05, at floor | healthy — embeddings look Gaussian |
| `pred` | → 0.005 | learning |
| loss | ~0.03 | low |
| `eff_rank` | **≈ 2** (of d=1024) | ⚠ |

Every standard check passed except the one added last. `tgt_std` is a **per-dimension marginal**
std averaged over dims: variance piled onto two directions still averages out fine. It is blind to
cross-dimensional correlation. Effective rank (participation ratio, `tr(C)² / ‖C‖_F²`) is not,
and it said the 1024-dim embedding lived in a ~2-dim subspace.

The probe was the arbiter: **Ω_m R² = 0.23**, σ8 = 0.09. Real signal, weak. The probe head's
validation loss flattened by epoch 3 — so the ceiling was the *features*, not the probe.

### The tempting wrong explanation
CAMELS has only ~2 recoverable degrees of freedom (the supervised benchmark shows the 4
astrophysical params carry 80–195% relative error even *with* labels). So "eff_rank ≈ 2 is
matched to the recoverable DoF, not broken" was an attractive story. It was wrong. Rank 2 was
the two strongest **nuisance** directions, and R²=0.23 was the symptom.

## 2. Root cause: SIGReg's gradient, not its value

The question isn't whether SIGReg *registers* low rank — it's whether it **pushes back**.
Measured against the real `sigreg_loss` at the true shape (B·n = 16384, d = 768), using a
rank-2 variance-matched blob vs an isotropic one:

| quantity | rank-2 blob | isotropic |
|---|---|---|
| SIGReg **value** | 0.035 | 0.00002 |
| SIGReg **‖∇‖** | **≈ 2e-4** | — |
| VICReg off-diag covariance **‖∇‖** | **≈ 1.25** | — |

SIGReg does register the collapse (0.035 ≫ 0.00002) but its gradient at the collapsed point is
**≈ 2e-4** — the same order as the prediction loss's shrink driver (4e-4). The covariance penalty
gives a gradient ~**6000×** stronger. SIGReg registers and cannot act.

**Why:** SIGReg is a mean of *marginal* Cramér–Wold tests over random projections. Low rank with
correctly-spread variance hides in the **correlation between** projections — which a
mean-of-marginals statistic never measures. Raising `n_proj` 256 → 512 did not help, confirming
this is structural, not sampling noise.

This is the **same blind spot** documented on Day 2 in
[`../analysis/sigreg_freq_dim_ablation.py`](../analysis/sigreg_freq_dim_ablation.py): random
projections of high-dim data are ~Gaussian by CLT / Diaconis–Freedman, so per-axis structure
survives only in the joint. The Day-2 ablation predicted this failure before it was observed in
training.

## 3. The five fixes

1. **VICReg variance-hinge + off-diagonal covariance** (`sigreg.py::variance_covariance_reg`) —
   supplies the non-vanishing, correctly-directed gradient SIGReg lacks. `relu(1 − std_j)` plus
   mean-squared off-diagonal covariance.
2. **Target LayerNorm** (`--target-norm`) — no-affine LayerNorm on pred & tgt before smooth-L1.
   Removes the collapse *driver*: shrinking toward a constant no longer lowers the loss.
3. **Multi-block masking** (`--n-blocks`, default 4) — target ratio 6% → 22%; a harder task
   demands richer features.
4. **Exact periodic augmentation** (`fields.py::_augment`) — roll/rot90/flip. The CAMELS box is
   periodic, so a circular shift is a *true* translation: no interpolation, no edge artifacts,
   and (Ω_m, σ8) are invariant → labels untouched.
5. **Attentive probe pool** (`probe.py::AttentivePool`) — learnable-query attention replaces
   mean-pooling (DINOv2 / I-JEPA eval standard). Encoder stays frozen → still an honest test.

## 4. Tuning: the ratio was inverted

The fixes alone did not work. Three 300-step gates:

| gate | var_coef | cov_coef | eff_rank | tgt_std | verdict |
|---|---|---|---|---|---|
| G1 | 1e-2 | 2e-2 | 1.8 → ~4 | **0.46 → 0.20** | soft fail — rank barely moved, scale collapsed |
| G2 | 1e-1 | 2e-2 | still 3–4 | — | loss flat; rank unmoved |
| G3 | 1e-1 | **2e-1** | **→ 1.0** | **→ 0.04** | hard fail — shrink-to-zero, killed at step 80 |

G2 and G3 together isolate the mechanism:

> **`var_coef` is the SCALE knob (fixes `tgt_std`); `cov_coef` is the RANK knob (fixes
> `eff_rank`).** Bumping var alone cannot move rank (G2). Bumping cov *without* a dominant var
> makes it worse (G3).

G3 is the instructive failure. There are two ways to drive off-diagonal covariance to zero:
decorrelate the dimensions, or **shrink every embedding toward the origin**. Shrinking is easier.
With cov ≥ var, the model took the cheap route: cov → 0.006 (near-perfect!) while `tgt_std` → 0.04
and rank hit the floor at 1.0. A near-perfect covariance score on a maximally collapsed model.

Root cause: **canonical VICReg runs var ≈ 25× cov.** We had run cov ≥ var throughout. The fix is
to make var dominant so it pins std ≈ 1 *first*, and only then let gentle cov decorrelate
*within* that maintained scale.

## 5. The winning recipe

```bash
--var-coef 5.0 --cov-coef 4e-2 --target-norm     # ~125:1, VICReg territory
```

300-step gate: eff_rank 1.8 → **28.5**, tgt_std → 0.91, var → 0.10, pred settled ~0.32
(genuine learning, not trivially satisfied), reg → 0.039. All monotone.

**1000-step keeper** (`ckpt.pt`, 2×A40, ~44 min, world=2, 22.7 GB/GPU, MFU 7%):

| metric | before (λ=0.02 collapsed) | after (winning recipe) |
|---|---|---|
| `eff_rank` | ~2 | **38.7** (peaked 40.6, still climbing) |
| `tgt_std` | 0.001 | **0.96** rock-steady |
| `pred` | 0.0001 | 0.256 (still descending) |
| **probe Ω_m R²** | **0.23** | **0.50** ⭐ **2.2×** |
| probe σ8 R² | 0.09 | 0.31 |

Effective rank 2 → 38 translated **directly** into probe signal. Ω_m > σ8 is the correct
recoverability ordering. Coverage 0.56 (vs 0.68 ideal) = σ slightly overconfident; the
calibration term tightens with training. Nothing plateaued → undertrained, not capped.

Baselines, honestly: supervised CMD `o3_err` CNN gets physical RMSE Ω_m 0.025 / σ8 0.045 (we're
~3× off, expected for a frozen label-free probe lacking the supervised CNN's circular-padding and
8× rot/flip inductive biases); the published VAE linear probe gets R² 0.93 (trained far longer).

## 6. Two lessons worth carrying

### Total loss is not a health metric
The collapsed λ=0.02 run had **loss 0.008**. The healthy run has **loss 1.2**. Loss went **up
150×** while the model got dramatically healthier — because loss is now ~92% the cranked var/cov
terms. Earlier, collapse was *cheaper* than honest features: at λ=0.02, collapsing gave
`0.98·0 + 0.02·0.4 = 0.008` vs ~0.196 for honest features — a 25× discount on collapse, which is
exactly what the optimizer took.

> Watch **components** (`eff_rank`, `tgt_std`, `var`, `pred`, `reg`). Never total loss. Never
> compare loss across configs.

### Why `cov` stays high (~21) yet rank climbs
`cov` was tiny (0.006) only when embeddings were shrunk to zero. With scale restored (std ~0.9),
covariances are naturally larger. Rank rises via the **var hinge**, not cov. High cov *drags rank
down* (it's in the denominator of the participation ratio), so 38 is achieved **despite** it —
which makes `cov_coef` safe headroom to raise later (var 5.0 pins the scale, so there's no
shrink risk) if the probe wants more. Not needed yet.

## 7. Open: is the *pooled* rank the real ceiling?

`eff_rank` is measured on `full_flat` — every patch token of every image, flattened. **The probe
never sees that cloud**: it pools each image's tokens into one vector per map. CAMELS fields are
spatially smooth, so intra-image tokens are highly redundant — the token cloud can carry rank 38
while the **pooled per-image vectors that actually feed the probe** carry far less. That number
has never been measured.

This matters before spending ~7 h on a 10k-step keeper: if pooled rank is the binding constraint,
more steps raise a token-level rank the probe never consumes, and the fix is to regularize the
**pooled** representation — a code change, not compute.

[`../../scripts/rank_report.py`](../../scripts/rank_report.py) settles it on a frozen checkpoint
with no training: token rank vs pooled rank, the pooled PCA spectrum, and a closed-form ridge
probe on the top-k PCs (sim-level split) to show how many dims carry *cosmology* rather than
nuisance variance.
