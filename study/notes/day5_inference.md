# Day 5 — Inference optimization (Stage 4)

**Target:** the project's *own* ViT-L encoder (`jepa_loss.ViTEncoder`, keeper config
`img=256 patch=16 d=1024 heads=16 layers=24`, 256 tokens/image), single A40, forward-only,
`eval()` + `no_grad`. Not the HuggingFace `infer.py`.

**Thesis carried over from Day 4:** the training harness is comms/memory-healthy but MFU
plateaued ~26% because `nn.TransformerEncoderLayer` dispatches an *unfused* attention path and
the graph is dispatched op-by-op from Python. Day 5 attacks the two remaining levers —
**fused attention (SDPA/FlashAttention)** and **graph capture (`torch.compile`)** — plus the
precision and batching levers, and measures each one in isolation with logged latency + throughput.

**Metrics per lever:** latency p50 / p99 (ms, per forward), throughput (images/sec),
peak mem (GB), MFU (%) against A40 bf16 peak 150 TFLOPS. Accuracy levers additionally report
cosine drift of the pooled embedding vs the fp32-eager reference.

---

## Predict before you measure (P1–P5)

Fill these in **before** running `scripts/bench_infer.py`. Then diff against the results table.
The point is to calibrate intuition, not to be right — write the number and the reason.

- **P1 — Baseline.** Eager, fp32, batch 64, ViT-L, one A40. What throughput (images/sec),
  order of magnitude? What MFU do you expect vs the ~26% you saw in *training* (remember: this is
  forward-only, no backward, no optimizer)?
  - prediction: _______
  - reason: _______

- **P2 — Precision (bf16 autocast).** Speedup factor over P1? (Day 4 saw 3.4× on the training
  loop when fp32→bf16 engaged the tensor cores. Forward-only — same, more, or less?)
  - prediction: _______
  - reason: _______

- **P3 — Fused attention (SDPA / FlashAttention).** On top of bf16, forcing the FlashAttention
  SDPA kernel — how much does it move throughput at 256 tokens? (Flash's win grows with sequence
  length; 256 is short. Big lever or small here?)
  - prediction: _______
  - reason: _______

- **P4 — `torch.compile`.** Over the bf16 baseline, what does graph capture + kernel fusion buy?
  Does it *stack* with the bf16 win or overlap with it? Any warmup / recompile cost to call out?
  - prediction: _______
  - reason: _______

- **P5 — Batching + int8 PTQ.**
  - (a) Sweeping batch 1 → 8 → 64 → 256: where does throughput saturate, and where does p99
    latency start climbing? Which regime is latency-bound vs throughput-bound?
  - (b) int8 dynamic PTQ on the `nn.Linear` layers — speedup, and what cosine drift on the pooled
    embedding would you accept before calling it too lossy?
  - prediction: _______
  - reason: _______

---

## Results

ViT-L (img256/patch16/d1024/heads16/L24, 256 tokens), batch 64, 1× A40, forward-only,
50 iters / 20 warmup. `torch.compile(mode="max-autotune")` fell back to default GEMMs
("not enough SMs" — Triton max-autotune templates didn't engage on this GPU).

| lever         | latency p50 (ms) | latency p99 (ms) | images/sec | peak mem (GB) | MFU (%) | cosine drift |
|---------------|------------------|------------------|-----------|---------------|---------|--------------|
| baseline      | 991.87           | 1039.26          | 64.4      | 1.32          | 4.6     | –            |
| bf16          | 262.49           | 264.54           | 244.0     | 1.19          | 17.3    | 1.87e-05     |
| bf16+flash    | 264.75           | 265.95           | 241.9     | 1.19          | 17.1    | 1.87e-05     |
| bf16+compile  | 213.20           | 215.17           | 300.5     | 0.84          | 21.3    | –            |
| int8 (CPU)    | 25801.49         | 28569.67         | 2.5       | n/a           | 0.2     | 3.39e-05     |

### Observations
- **bf16 is the dominant lever (P2).** 64 → 244 img/s = **3.8×** over the fp32 baseline, MFU
  4.6 → 17.3%, for a cosine drift of 1.87e-5 (five digits preserved — effectively free). Tensor
  cores replacing CUDA-core fp32 matmuls, same mechanism as the Day-4 training sweep (3.4× there).
- **Flash is a no-op at 256 tokens (P3 — predicted).** bf16+flash = 242 img/s vs bf16's 244, drift
  identical to the digit. autocast already disables `nn.TransformerEncoderLayer`'s fused C++ path and
  routes to `F.scaled_dot_product_attention`, where PyTorch auto-selects a fused backend anyway —
  so *forcing* FLASH changes nothing. Flash's win scales with sequence length; 256 tokens is too
  short for it to matter.
- **compile stacks on bf16 (P4).** 244 → 300 img/s = **+23%** (4.7× over baseline total), MFU 21.3%,
  and peak mem 1.19 → 0.84 GB — graph fusion removes intermediate activation allocations. And this is
  the *un-tuned* number: `max-autotune` GEMM never engaged ("not enough SMs"), so there's likely more
  on the table with CUDA graphs / a GPU where the Triton templates fire.
- **int8 dynamic PTQ is a CPU tool, not a GPU-latency lever (P5b).** 26 s/batch, ~120× slower than
  bf16-GPU — because `quantize_dynamic` executes on CPU. Its cosine drift (3.4e-5) is tiny, so the
  *accuracy* cost is negligible; the point is that the speed story only exists for CPU deployment.
  GPU int8 would need static quant / TensorRT hitting the int8 tensor cores.
- **MFU ceiling ~21%, *below* training's 26% (Day 4).** Counterintuitive but instructive: forward-only
  inference has 3× lower arithmetic intensity than the training step (`2·N·tokens` vs `6·N·tokens`),
  so kernels are shorter and more launch/memory-bound — less compute to amortize dispatch over. The
  remaining headroom is throughput-side: **larger batch** (amortize launches) and **CUDA graphs /
  working max-autotune**, not precision. → run the batch sweep (P5a) to find where MFU saturates.

### P5a batching sweep (bf16+compile)

`python scripts/bench_infer.py --lever bf16+compile --batch-sweep 1,8,64,256`

| batch | latency p50 (ms) | latency p99 (ms) | images/sec | peak mem (GB) | MFU (%) |
|-------|------------------|------------------|-----------|---------------|---------|
| 1     | 6.73             | 6.86             | 148.3     | 0.83          | 10.5    |
| 8     | 117.57           | 118.05           | 68.0      | 0.83          | 4.8 †   |
| 64    | 208.60           | 210.73           | 307.1     | 0.84          | 21.8    |
| 256   | 805.01           | 825.66           | 318.2     | 0.89          | 22.5    |

- **Throughput saturates by batch 64.** b64 → b256 quadruples batch and latency (209 → 805 ms) for
  +3.6% throughput (307 → 318 img/s); MFU flatlines at ~22%. The binding constraint is NOT batch
  size — it's kernel efficiency (`max-autotune` GEMM never engaged; launch-bound). To push past ~22%
  the lever is CUDA graphs / working max-autotune, not more batch.
- **Latency vs throughput tradeoff, measured.** b1 = 6.7 ms latency (real-time/interactive) but only
  148 img/s at 10.5% MFU (GPU underfed); b256 = 318 img/s (throughput-optimal) at 805 ms latency
  (offline batch only). Peak mem barely moves (0.83 → 0.89 GB) — activations are cheap at 256 tokens,
  so batch is a free throughput knob until the kernels saturate.
- † **b8 is a REPRODUCIBLE per-shape slowdown, not noise.** Isolated re-run
  (`CUDA_VISIBLE_DEVICES=0 ... --batch 8 --warmup 20`) reproduced it to the digit (117.4 ms,
  68.1 img/s, MFU 4.8%), so the initial co-tenant-interference guess was wrong. b8 runs ~2× slower
  per image than b1 and ~4.5× slower than the b64/b256 regime, and the kernel is stable (p50/p99
  within 0.4 ms — not recompilation thrash). Most likely an **Inductor kernel-selection cliff** for
  this shape: `max-autotune` GEMM is disabled ("not enough SMs"), so `torch.compile` falls back to a
  default tile-size heuristic that picks a poor config at M = 8·256 = 2048 while M ∈ {256, 16384,
  65536} land well. Discriminator: run `--lever bf16 --batch-sweep 1,8,64,256` (no compile); if b8 is
  on-trend there, the cliff is compile/Inductor, not hardware GEMM tiling. [discriminator pending]
