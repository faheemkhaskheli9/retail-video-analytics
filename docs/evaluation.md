# Evaluation Notes: Retail Video Analytics Platform

## Metrics

No public labeled retail-CCTV dataset with entry/exit and dwell-time ground
truth ships with this project (see README §8), so evaluation instead uses
the synthetic scene in `retail_video_analytics/synthetic.py`, which has exact
ground truth by construction (3 actors, known trajectories, known dwell
times). The metric reported is **unique-customer counting accuracy**:
`|counted_unique_customers - true_actor_count|`, measured as the upstream
detector's simulated noise (Gaussian box jitter) and miss rate increase. This
directly probes the tracker's weakest point -- it is a greedy IoU matcher
with no motion model or re-identification, so noisier/missing detections are
expected to fragment a single actor's track into several IDs.

## Reproducing Results

```bash
python -m retail_video_analytics.evaluate --config configs/eval.yaml
```

## Result Log

| Date | Config | Metric | Value | Notes |
|------|--------|--------|-------|-------|
| 2026-08-20 | eval.yaml, scenario=clean (noise_std=0, miss_rate=0) | unique-customer counting error | 0 / 3 | Exact match: no detector noise, tracker never fragments a track. |
| 2026-08-20 | eval.yaml, scenario=moderate_noise (noise_std=3px, miss_rate=0.05) | unique-customer counting error | 4 (7 counted vs 3 true) | Track fragmentation begins: brief missed detections plus box jitter push IoU below the match threshold and spawn new track IDs. |
| 2026-08-20 | eval.yaml, scenario=high_noise (noise_std=8px, miss_rate=0.2) | unique-customer counting error | 20 (23 counted vs 3 true) | Severe fragmentation. Confirms `IOUTracker`'s documented limitation (no Kalman filter / re-ID) -- a production deployment should use ByteTrack or DeepSORT, as scoped in the README's tech stack, particularly under detector noise this high. |

`max_concurrent_customers` (a simpler, track-fragmentation-insensitive
metric) stayed correct at 3 across all three scenarios in every run, since it
only depends on how many boxes are active in a given frame, not on ID
continuity across frames.

## Interpreting these numbers

These are internal consistency checks against synthetic ground truth, not a
benchmark against a public tracking dataset (e.g. MOT17) -- there was no
labeled real-world retail footage available for this portfolio build. Treat
the counting-error trend (worse tracking under noisier detections) as the
finding, not the exact error magnitudes, which are specific to this
synthetic scene's actor geometry and the noise model in
`synthetic.SimulatedDetector`.
