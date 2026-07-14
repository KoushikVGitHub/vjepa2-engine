# Day 0 — Study Map: The State of World Models (June 2026)

Landscape primer. Goal: be conversant in the *whole* world-model field, not just JEPA —
so you sound like an insider in the AMI interview, not a fan.

> Read actively. At each claim ask *"why this and not the alternative?"* (elaborative
> interrogation). Sketch the two-camp split by hand (dual coding).

---

## The two paradigms — the real axis is *what you predict*

Both camps build "world models" (a model that predicts how the world evolves, optionally
conditioned on actions). The split is **what space the prediction happens in**:

### 1. Compress-to-understand — predict in latent / representation space  ← AMI / LeCun
- Encode input into an abstract representation, then predict the *representation* of the
  future, **not the pixels**.
- Family: **JEPA → I-JEPA → V-JEPA → V-JEPA 2**, **Dreamer** line,
  **LeWorldModel** (Mar 2026, end-to-end from pixels but still latent-predictive).
- An **energy-based** view: low energy when the predicted representation matches the actual
  future representation, high when it mismatches.

### 2. Render-to-predict — generate pixels
- Predict the actual future *frames* — a generative video model, often diffusion or
  autoregressive-over-tokens.
- Family: **Genie 3** (photorealistic interactive worlds, ~24fps), **Dreamer 4**,
  **GAIA-2** (driving), Sora-style video, Alibaba **Happy Oyster** (Apr 2026).

> Note: "generative vs energy-based" is the right *axis*, but **both are world models**.
> The clean framing is *pixel/generative prediction* vs *latent/energy-based prediction*.

---

## Why AMI bets on compression (the case)

One-liner: **predicting pixels forces the model to waste capacity modeling unpredictable,
irrelevant detail** (every leaf, texture, sensor noise). Predicting in *representation
space* lets the model **throw away what it can't and shouldn't predict** and keep only the
abstract structure that matters for understanding and **planning**.

LeCun's claim: that abstraction is the actual road to human-like reasoning/AGI, whereas
autoregressive token/pixel prediction (LLMs, video diffusion) is a detour.

Payoff — **planning = energy minimization**: roll candidate futures forward in latent space
-> a **cost module** scores each -> a planner executes the first step of the minimum-cost
plan, then re-plans (MPC).

> ★ Carry this thread: the **cost module is the alignment surface** — your safety
> differentiator. Controllability/safety are *specified as the energy/objective*.

---

## The strongest argument *against* the bet (June 2026)

Not primarily "latent error is unmanageable" (that's the Day-5 failure-mode topic). The
*strategic* counterargument:
- **Generative world models are shipping at scale and look more impressive** — Genie 3 gives
  interactive photorealistic worlds *today*; JEPA-at-scale is thin
  (**V-JEPA 2 ~= one model, one lab**).
- **No proof the compression path scales to AGI** — it wins on *control tasks* in head-to-head
  MPC loops, but the flashy, fundable, productized momentum is on the rendering side.

Bet summary: **JEPA-style models beat video-diffusion world models on control in the same
planner — but generative is winning on scale, capital, and mindshare.** AMI bets compression
is the real road despite rendering looking better right now.

---

## Interview line to keep (Day 0 deliverable seed)
*"Why does AMI bet on compression when generative world models look more impressive?"*
-> 3-sentence answer goes here after the post-test, then into the README context section.

---

### Sources
- LeCun (2022), *A Path Towards Autonomous Machine Intelligence* — OpenReview.
- Meta (2025), *V-JEPA 2 world model and benchmarks* — ai.meta.com/blog.
- Field scan, June 2026: Genie 3, Dreamer 4, GAIA-2, LeWorldModel, Happy Oyster.
