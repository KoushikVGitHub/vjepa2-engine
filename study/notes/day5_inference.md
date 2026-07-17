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

_(run the benchmark, paste the markdown table it prints here, then write the observations —
lever-by-lever, in the Day-4 style: what moved, by how much, and the mechanism.)_

| lever | dtype | compile | latency p50 (ms) | latency p99 (ms) | images/sec | peak mem (GB) | MFU (%) | cosine drift |
|-------|-------|---------|------------------|------------------|-----------|---------------|---------|--------------|
| _tbd_ |       |         |                  |                  |           |               |         |              |

### Observations
- _tbd_
