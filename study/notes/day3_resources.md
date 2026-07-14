# Day 3 resources — Video & scale: V-JEPA 2 + data-curation pipeline

Day-3 build = decode -> clip-sample -> resize/normalize -> tubelet-ify -> batched loader,
then PROFILE clips/sec and identify the bottleneck (decode? IO? transform?).
Curation is Koushik's STEM (scorecard box 5 — the AMI JD bullet to land for real), so go
deepest on Strand A.

## Strand A — Video data curation at scale (the stem — heaviest weight)
- LAION `video2dataset` — https://laion.ai/blog/video2dataset/
  THE reference: real large-scale curation tool (built a 590M video-text set). Sharding,
  distributed download/decode, transforms = the vocabulary that makes "curation at scale" real.
- Encord — Video Data Curation Guide — https://encord.com/blog/data-curation-guide-for-video/
  The decisions layer: scene-cut detection, optical-flow static/dynamic filtering, near-dup
  removal. Directly answers Day-3 post-test (design a filter to drop low-info/static/dup clips).
- Breaking the Bottleneck: GPU-Optimised Video Processing (TDS) —
  https://towardsdatascience.com/breaking-the-bottleneck-gpu-optimised-video-processing-for-deep-learning/
  Throughput/profiling (decode vs IO vs transform). Read BEFORE profiling so you predict the
  bottleneck (Day-3 P4) then check.
- Vid Prepper (TDS) — https://towardsdatascience.com/introducing-vid-prepper/
- Preparing Video Data for Training: A PyTorch Guide (Medium) —
  https://medium.com/@naneettyagi2004/preparing-video-data-for-training-a-pytorch-guide-fc644ee9e64c
  Hands-on FFmpeg/NVDec + PyTorch Dataset mechanics for the build.

## Strand B — Tubelet / spatiotemporal masking (images -> video)
- VideoMAE explainer (Medium) —
  https://medium.com/@kdk199604/videomae-scaling-self-supervised-video-transformers-beyond-labeled-data-1048a780522d
  Key idea: ONE mask pattern across all frames ("tube") at 90-95% ratio, because temporal
  correlation gives masked patches an unmasked twin in adjacent frames -> info leakage ->
  trivial task. That "why tubes, why so aggressive" = Day-3 P1.
- VideoMAE paper — https://ar5iv.labs.arxiv.org/html/2203.12602
- VideoMAE V2 (dual masking, scaling) — https://arxiv.org/pdf/2303.16727

## Strand C — V-JEPA 2 (model + recipe)
- Meta — Introducing V-JEPA 2 — https://ai.meta.com/research/vjepa/  (authoritative; 1M-hr
  video + 62-hr robot recipe, planning story)
- LearnOpenCV — V-JEPA 2 World Model for Robotics —
  https://learnopencv.com/v-jepa-2-meta-world-model-robotics-guide/  (figure/video-driven)
- The Annotated JEPA — https://elonlit.com/scrivings/the-annotated-jepa/  (line-by-line)
- Gonzo ML — V-JEPA 2: Scaling V-JEPA — https://gonzoml.substack.com/p/v-jepa-2-scaling-v-jepa
- Repo (reference data pipeline) — https://github.com/facebookresearch/vjepa2

## Loose end to close (Day-1 build skipped)
- LearnOpenCV — Optimizing VJEPA-2 Real-Time Classification —
  https://learnopencv.com/optimizing-vjepa-2-in-real-time-video-classification/
  Working inference script = the Day-1 BUILD (run inference on clips, extract embeddings,
  cosine-sim). Cheap to knock out alongside Day 3.

Lightest passive viewing: LearnOpenCV (A & C) + The Annotated JEPA. LAION + Encord are
better as ACTIVE reading while building.
