# Architecture Notes: Retail Video Analytics Platform

## Status

Phase 1 (detection + tracking pipeline) and Phase 2 (zone definitions +
dwell-time / cashier-presence logic) are implemented and covered by tests.
Phase 3 (heatmap persistence/visualization) is partially implemented (raw
heatmap samples are collected in-memory; no rendering/storage yet). Phase 4
(operator dashboard) is not implemented.

## Pipeline

```text
Camera Feed / Synthetic Scene -> Detector -> IOUTracker -> ZoneManager + EventEngine -> AnalyticsStore (SQLite)
```

Implemented as `retail_video_analytics.pipeline.VideoAnalyticsPipeline.run()`,
which pulls frames from either a real video file (`cv2.VideoCapture`) or the
synthetic demo scene, runs the configured detector, updates the tracker,
feeds resulting tracks into the zone/event engine, and persists a summary.

## Components

| Component | Module | Notes |
|---|---|---|
| Detection | `detection/hog_detector.py` | Default backend: OpenCV's built-in HOG + linear-SVM people detector. Ships with `opencv-python-headless`, needs no model download, runs on CPU. Real, non-mocked CV algorithm. |
| Detection (optional) | `detection/yolo_detector.py` | Ultralytics YOLO backend behind the same `Detector` interface, for higher accuracy on real footage. Guarded/optional import; requires `pip install ultralytics` and a network fetch of weights the first time it runs. Exercised in tests via dependency injection (a fake model object), not a real network call. |
| Tracking | `tracking/tracker.py` | `IOUTracker`: a simplified SORT (greedy IoU matching + track aging, no Kalman filter/appearance re-ID). Deliberately simple for an MVP; see Limitations. |
| Zones | `zones/zone.py` | Polygon zones from YAML config, ray-casting point-in-polygon test. `kind` (`entrance` / `checkout_counter` / `generic`) drives event semantics. |
| Zone/event logic | `events/events.py` | `EventEngine`: entry/exit counting for `entrance`-kind zones, per-track dwell-time accumulation for every zone, and cashier/counter-absence alerting for `checkout_counter`-kind zones (fires once an empty streak crosses a configurable duration, resets on presence). |
| Storage | `storage/db.py` | `AnalyticsStore`: SQLite (stdlib `sqlite3`) instead of the README's PostgreSQL, since standing up a Postgres instance is out of scope for a portfolio MVP. Same repository-style API, so swapping the connection/driver later doesn't touch calling code. |
| Synthetic scene | `synthetic.py` | Generates a small scene (3 actors on straight-line walks between an entrance and a checkout area) with known ground truth, used for the offline demo and for evaluation. `SimulatedDetector` replays that ground truth with configurable Gaussian noise and a miss rate to approximate a real detector's imperfection when no camera feed is available. |
| CLI | `main.py`, `evaluate.py` | `python -m retail_video_analytics.main --config configs/default.yaml` runs the pipeline once and prints/persists a JSON summary. `python -m retail_video_analytics.evaluate --config configs/eval.yaml` runs the synthetic scene through three detector-noise scenarios and reports counting accuracy for each. |

## Design Notes

- Detection is behind a small `Detector` ABC (`detection/base.py`) so
  tracking/zone/event/storage code never depends on which backend produced a
  box. `configs/*.yaml`'s `detector.backend` selects `hog` (default, offline)
  or `yolo` (optional, needs network + `ultralytics`).
- Configuration-driven: zones, detector/tracker parameters, cashier-absence
  threshold, and storage path all come from YAML in `configs/`, not
  hardcoded constants.
- The two unavailable-in-this-environment boundaries -- a real network model
  download for YOLO, and real retail CCTV footage -- are both handled
  explicitly rather than silently skipped: YOLO is dependency-injected in
  tests (`tests/test_detection.py::test_yolo_detector_uses_injected_model_without_network`),
  and the synthetic scene generator stands in for a camera (`synthetic.py`,
  `tests/test_synthetic.py`).

## Known Limitations (see also README §15)

- `IOUTracker` has no motion model (Kalman filter) or appearance re-
  identification. It works well on the synthetic scene (well-separated, low-
  noise detections) but the evaluation run
  (`python -m retail_video_analytics.evaluate`) shows unique-customer counts
  inflating sharply as detector noise/miss-rate increase -- see
  `docs/evaluation.md`. This is an expected, documented weakness of a greedy
  IoU tracker, not a bug: production systems use ByteTrack/DeepSORT (as
  listed in the README's tech stack) specifically to address it.
- `HOGPersonDetector` is a classical, comparatively low-accuracy detector
  (no learned features beyond a linear SVM over HOG descriptors). It is used
  as the offline default so the pipeline is runnable without any download;
  swap `detector.backend: yolo` for real deployments.
- Heatmap rendering and the operator dashboard (Phases 3-4) are not built;
  `EventEngine.heatmap_samples` collects the raw centroid points a heatmap
  would be built from, but nothing renders or persists them yet.
