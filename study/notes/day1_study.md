# Day 1 — Study Briefing: Why world models, and the energy-based view

Concept: LeCun's AMI vision; predicting in **latent space** vs generating pixels/tokens;
energy-based models; representation **collapse** = the central problem.

> Read actively: at each claim ask *"why this and not the alternative?"* Sketch the JEPA vs
> VAE vs SimCLR comparison by hand.

## 1. Why pixel prediction is a bad objective
Predicting future pixels forces the model to predict the **unpredictable and the irrelevant**
— exact textures, leaf positions, sensor noise. That detail is impossible to predict AND
useless for deciding what to do, so the model burns capacity modeling noise. Predicting in
**representation space** lets the encoder *discard* unpredictable detail and keep only the
abstract structure (object identity, position, motion, dynamics) a planner needs.

## 2. Representation collapse — the central problem
**Collapse = the encoder maps everything to the same / a trivial constant representation.**
A naive JEPA falls in because if both encoders are trainable and you just minimize
"predicted latent vs target latent," the **degenerate shortcut** is to make every
representation a constant: predictor trivially outputs the constant, prediction error -> 0,
loss minimized, model has learned **nothing**.

Key correction: collapse is NOT "error too large" — it's the model **cheating the loss to
zero** by throwing away all information. The loss looks great; representations are useless.
Everything on Day 2 (EMA target, stop-gradient, asymmetry) exists to prevent this one failure.

## 3. How JEPA avoids collapse without contrastive negatives

| | What it predicts | How it avoids collapse | Decoder? |
|---|---|---|---|
| **VAE** | Reconstructs **pixels** | Reconstruction forces info retention | **Yes** (pixel decoder) |
| **SimCLR (contrastive)** | instance discrimination | **Negative pairs** push samples apart | No |
| **JEPA** | **Latent representation** of masked/future part | **Asymmetry**: EMA target + **stop-gradient** (no negatives) | **No** |

Insider point: JEPA is **non-contrastive AND non-generative** — no negative pairs (unlike
SimCLR), no pixel decoder (unlike VAE). It dodges collapse purely through **architectural
asymmetry**: the target encoder is an EMA copy with stop-gradient, so the target can't
trivially collapse to match the predictor.

## 4. The energy-based view (carry this thread)
JEPA **is an EBM over representations**: energy low when predicted representation matches the
actual target, high when it mismatches. Training pushes energy down on real (context, target)
pairs. The anti-collapse machinery stops the energy landscape going flat (= collapse). Later,
**planning = energy minimization**: roll futures -> cost module scores -> execute min-energy
plan -> re-plan.

## 5. JEPA vs LLMs / LeCun's AMI bet
LLMs are autoregressive **token** predictors — generative, discrete output space. LeCun:
that's a detour to AGI because (a) it predicts in output space (the pixel-prediction problem
in token form) and (b) no persistent world model or planning-by-energy-minimization. World
models that predict in **latent space** and plan via **energy** are, in his view, the
architecture for real reasoning/agency in the physical world.

## 6. P4 — prediction to verify in the build
Cosine similarity should be **higher for similar clips, lower for different clips** (a good
representation places similar inputs near each other).

---

### Sources
- LeCun (2022), *A Path Towards Autonomous Machine Intelligence* — OpenReview (JEPA + cost-module sections).
- Assran et al. (2023), *I-JEPA* — CVPR (architecture + anti-collapse).
- Meta (2025), *V-JEPA 2 world model* — ai.meta.com/blog.
