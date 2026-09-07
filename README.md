# Retail Video Analytics Platform

> Computer Vision & Video Analytics portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-mvp-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Retail operators want to understand customer flow, dwell time, and staffing coverage from existing camera feeds without manual observation.

## 2. Architecture

```text
Camera Feed / Synthetic Scene -> Detector (HOG default, YOLO optional) -> IOUTracker -> Zone/Event Logic -> SQLite -> (Dashboard: not yet built)
```

See `docs/architecture.md` for the full component breakdown.

## 3. Technology Stack

- Python
- OpenCV (`opencv-python-headless`) -- built-in HOG+SVM detector (default, offline) and video I/O
- Ultralytics YOLO -- optional higher-accuracy detector backend (extra dependency, needs network for weights)
- Custom IoU-based multi-object tracker (simplified SORT; ByteTrack/DeepSORT noted as future work, see Limitations)
- SQLite (MVP; README originally specified PostgreSQL for production -- see Limitations)
- FastAPI / React dashboard -- not yet built (Phase 4)

## 4. Feature List

- [x] Person detection (HOG default; YOLO backend behind the same interface)
- [x] Multi-object tracking (IoU-based, with track aging through brief occlusion)
- [x] Customer counting (unique + max-concurrent)
- [x] Entry/exit counting (per `entrance`-kind zone)
- [x] Dwell-time measurement (per track, per zone)
- [x] Zone definitions (YAML polygons, ray-casting membership test)
- [x] Customer heatmaps -- centroid samples binned into a density grid, rendered to PNG + JSON per run and re-generatable from a stored run (`retail_video_analytics.heatmap_cli`)
- [ ] Cashier/staff detection -- no separate staff-vs-customer classifier; see Limitations
- [x] Counter-presence detection (`checkout_counter`-kind zone occupancy)
- [x] Cashier-absence time tracking (threshold-based alert, configurable)
- [ ] Historical analytics dashboard -- not built (Phase 4)

## 5. Implementation Plan

1. **Phase 1 (done):** Detection + tracking pipeline on sample video (synthetic scene + a real-video code path via OpenCV).
2. **Phase 2 (done):** Zone definition tooling and dwell-time / entry-exit / cashier-absence logic.
3. **Phase 3 (partial):** Heatmap and historical analytics storage -- SQLite run storage and traffic/dwell heatmap rendering/persistence are done; ByteTrack/DeepSORT swap + re-eval is not.
4. **Phase 4 (not started):** Dashboard for store operators.

## 6. Repository Structure

```text
retail-video-analytics/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
│   └── retail_video_analytics/
│       ├── detection/       # Detector interface, HOG (default) and YOLO (optional) backends
│       ├── tracking/        # IoU-based multi-object tracker
│       ├── zones/           # Polygon zone definitions + membership test
│       ├── events/          # Dwell-time / entry-exit / cashier-absence logic
│       ├── storage/         # SQLite analytics store
│       ├── synthetic.py     # Synthetic demo scene + simulated-detector noise model
│       ├── pipeline.py       # Orchestrates detection -> tracking -> events -> storage
│       ├── main.py           # CLI: run the pipeline once
│       └── evaluate.py       # CLI: counting-accuracy evaluation vs synthetic ground truth
├── tests/
├── configs/
│   ├── default.yaml
│   └── eval.yaml
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd retail-video-analytics
uv venv --python 3.12 .venv        # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"         # or: pip install -r requirements.txt
cp .env.example .env               # not currently used by the pipeline; kept for future API/DB config
```

## 8. Dataset

No public retail-CCTV dataset with entry/exit and dwell-time ground truth
ships with this project. Instead, `retail_video_analytics/synthetic.py`
generates a small synthetic scene (3 actors walking straight-line paths from
an entrance toward a checkout area) with exact ground-truth positions every
frame, used for both the offline demo (`video_source: synthetic` in
`configs/default.yaml`) and for evaluation. The pipeline also has a real
code path (`_iter_video_frames`, via `cv2.VideoCapture`) for any real video
file the operator points it at -- no real footage is bundled here.
No proprietary, employer-owned, or client-identifiable data is used in this project.

## 9. Training / Execution

There is no training step (both detector backends are pre-trained / off the
shelf: OpenCV's built-in HOG+SVM, or Ultralytics YOLO). To run the pipeline:

```bash
python -m retail_video_analytics.main --config configs/default.yaml
python -m retail_video_analytics.main --config configs/default.yaml --video path/to/clip.mp4
```

## 10. Evaluation

```bash
python -m retail_video_analytics.evaluate --config configs/eval.yaml
```

See `docs/evaluation.md` for the metric definition, how to reproduce it, and
the current Result Log.

## 11. Results

- **What runs for real:** the full pipeline (detection -> IoU tracking ->
  zone/dwell/entry-exit/cashier-absence logic -> SQLite storage) executes
  end to end, on CPU, with no external services or downloads required, using
  either the built-in `HOGPersonDetector` on real video or the synthetic
  demo scene. 52 automated tests exercise every module (`pytest tests/`).
- **What's simplified/mocked:** there is no real labeled retail-camera
  footage in this environment, so the demo/evaluation input is a synthetic
  scene with known ground truth (`synthetic.py`), not real CCTV video. The
  optional YOLO backend is real integration code but is exercised in tests
  via a dependency-injected fake model, not a real network weights download.
  The tracker is a simplified greedy-IoU SORT variant (no Kalman filter, no
  appearance re-identification) -- `docs/evaluation.md` shows this degrading
  visibly under simulated detector noise. There is no separate cashier/staff
  classifier; "cashier presence" is inferred purely from occupancy of a
  `checkout_counter`-kind zone, so any tracked person standing there counts
  as staff present.
- **What's not built yet:** the operator dashboard (Phase 4) and the
  ByteTrack/DeepSORT tracker swap (Phase 3).

## 12. API

No HTTP API yet -- the pipeline currently runs as a CLI
(`retail_video_analytics.main`) that prints and persists a JSON summary. A
FastAPI layer over `AnalyticsStore` is future work (see §16).

## 13. Docker

Not yet added for this MVP (no API server to containerize yet -- see §12).
Once the Phase 4 dashboard/API exists, this will build and run it:

```bash
docker build -t retail-video-analytics .
docker run -p 8000:8000 retail-video-analytics
```

## 14. Tests

```bash
uv pip install -e ".[dev]"
uv run pytest tests/    # 52 tests, CPU-only, ~15s, no network/GPU required
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers are based on a synthetic scene, not real footage or a
  public tracking benchmark (e.g. MOT17) -- they demonstrate relative
  behavior (accuracy degrading under detector noise), not absolute
  real-world accuracy. See `docs/evaluation.md`.
- The tracker (`IOUTracker`) is a simplified greedy-IoU matcher with no
  Kalman filter or appearance re-identification, so it fragments tracks
  under detector noise/misses more than a production ByteTrack/DeepSORT
  tracker would (see the tech stack note in §3 and the evaluation results).
- Storage uses SQLite instead of the PostgreSQL named in the original tech
  stack, to keep the MVP dependency-free; `storage/db.py`'s repository-style
  API is designed so swapping the backend later doesn't touch calling code.
- The default `HOGPersonDetector` is a classical, comparatively low-accuracy
  detector chosen specifically because it requires no model download and
  runs anywhere `opencv-python-headless` installs. The `YoloDetector`
  backend behind the same interface is the intended higher-accuracy option
  for real deployments (`detector.backend: yolo` in config), but needs
  `pip install ultralytics` and a one-time network fetch of model weights
  neither of which this environment's test suite depends on.
- No separate cashier/staff detector -- "cashier presence" is inferred from
  any tracked person occupying a `checkout_counter`-kind zone.
- The operator dashboard (README Phase 4) and the ByteTrack/DeepSORT tracker
  swap (README Phase 3) are not implemented.

## 16. Future Work

- Swap the tracker for ByteTrack or DeepSORT and re-run the evaluation to
  quantify the accuracy improvement under the same noise scenarios.
- [x] Render/persist the heatmap samples `EventEngine` already collects (`python -m retail_video_analytics.heatmap_cli --run-id <id>`; also written automatically per persisted pipeline run into `storage.heatmap_dir`).
- Build the FastAPI + React operator dashboard (Phase 4).
- Expand evaluation coverage and add CI-based regression checks.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-08-20_
