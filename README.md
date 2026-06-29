# vjepa2-engine

**End-to-end engineering of a V-JEPA-style video world-model trainer:** data curation at scale → distributed training → optimized inference. A from-scratch, reproducible build of the *engineering pipeline* a self-supervised video model actually needs in production — the three things a world-model lab runs every day.

> Self-supervised video models live or die on infrastructure: how fast you can curate petabytes of video, how efficiently you shard a model across GPUs, and how cheaply you can serve it. This repo builds and benchmarks all three, on real video, with logged numbers.

Built as a focused engineering intensive. Method credibility (from-scratch JEPA + modern anti-collapse) is included so the infrastructure isn't a black box, but the headline is **systems, throughput, and cost** — not a new model.

## The pipeline (maps to how a video world model is built)

| Stage | File | What it does | Status |
|---|---|---|---|
| **1 · Curation at scale** | `src/data/curation.py` | Decode → clip-sample → motion-filter → normalized batched loader. Manifest-driven offline curation; profiled `clips/sec` and found the bottleneck *moves* under parallelism. | ✅ built + profiled |
| **2 · The method (from scratch)** | `src/jepa_loss.py`, `src/sigreg.py` | Minimal JEPA (masking, EMA target, stop-grad predictor) + SIGReg anti-collapse — so the training loop isn't a black box. Empirically reproduced *stop-gradient*, not EMA decay, as the load-bearing anti-collapse mechanism. | ✅ built + verified |
| **3 · Distributed training** | `src/train_fsdp.py` | Wrap the JEPA loop in **PyTorch FSDP/DDP**, bf16 mixed precision, activation checkpointing. Real 2×A40 run, throughput + memory + MFU logged per lever ([`notes/day4_results.md`](notes/day4_results.md)). | ✅ built + benchmarked |
| **4 · Inference optimization** | `src/infer.py` | Baseline encoder inference → `torch.compile`, bf16/fp16, batching, optional int8 PTQ. Latency (p50/p99) + throughput benchmark, lever-by-lever. | 🔜 Day 5 |

**Why this scope:** it mirrors the three production-engineering competencies a self-supervised video lab needs — *curate → train at scale → serve fast* — built end to end rather than as disconnected demos.

## Context — where this sits (June 2026)
World-model research splits by *what space prediction happens in*. **Render-to-predict** models generate future frames (Genie 3, Dreamer 4, GAIA-2, Sora-style) — impressive photorealism, heavy compute. **Compress-to-understand** models — JEPA / **V-JEPA 2**, LeWorldModel — predict in *latent space*, discarding unpredictable pixel detail and keeping the abstract structure needed for planning. JEPA-style latent models already match or beat video-diffusion world models on control tasks in the same planning loop, at a fraction of the inference cost — which is exactly why the *engineering* (curation throughput, distributed training, serving efficiency) is the bottleneck worth being excellent at. This repo works inside the compression camp and optimizes that engineering surface.

## Highlights (the parts worth reading)
- **Curation bottleneck *moves* under parallelism** — serial decode is decode-bound, but at 8 workers the cost shifts to framework plumbing (worker spawn + fp32-tensor IPC to a single main process). Lever: ship **uint8 over IPC (4× smaller), normalize on GPU**. Measured, not assumed.
- **Curation is a *policy*, not a constant** — rejection threshold read off the empirical motion distribution's lower tail (~p10) so only genuine dead-air is dropped; quantified rejection rate and its hidden ~1.8× resample cost → argues for *offline* curation, not per-`__getitem__`.
- **SIGReg is distributed-friendly by construction** — its anti-collapse regularizer is an expectation over the batch, so at scale you just **all-reduce per-GPU partial statistics** for the global-batch statistic; no cross-device negative-pair gathering (contrast: SimCLR). A clean primitive for distributed world-model training.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m src.data.curation        # profile the curation loader on samples/
```

## Results (logged as built)
- [x] **Curation pipeline (Stage 1):** decode→filter→loader on real video; rejection rate quantified; bottleneck characterized (decode-bound serial → plumbing-bound at 8 workers); threshold set from motion distribution tail.
- [x] **Method reproduction (Stage 2):** from-scratch JEPA collapse study — stop-gradient (not EMA decay) is load-bearing; symmetric variant collapses (std→0, loss→0) as predicted.
- [x] **Distributed training (Stage 3):** 2×A40, 114.5M-param JEPA. DDP/fp32 baseline → FSDP+bf16 = **146 → 492 samples/sec (3.4×)**, MFU 7.7% → 25.7%; adding activation checkpointing cuts peak memory **12.1 → 1.6 GB (7.5×)** at ~20% throughput cost. bf16 is the throughput lever (engages tensor cores), checkpointing the memory lever; FSDP-vs-DDP is throughput-neutral at this scale. Full table + analysis: [`notes/day4_results.md`](notes/day4_results.md).
- [ ] **Inference optimization (Stage 4):** latency p50/p99 + throughput per lever (`torch.compile`, bf16, batching, int8) — baseline `___` → optimized `___`.

## Status
Active engineering build. Not affiliated with Meta or AMI Labs. Repo demonstrates production-engineering skills for self-supervised video world models: large-scale data curation, distributed training, and inference optimization.

— [github.com/KoushikVGitHub/vjepa2-engine](https://github.com/KoushikVGitHub/vjepa2-engine)
