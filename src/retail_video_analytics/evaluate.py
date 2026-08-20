"""Lightweight self-evaluation against the synthetic ground-truth scene.

There is no public labeled retail-CCTV dataset bundled with this project (see
README section 8), so evaluation here measures the pipeline's counting
accuracy against synthetic ground truth under increasing simulated detector
noise/miss rates -- i.e. "how much does tracking-and-counting accuracy
degrade as the upstream detector gets worse", which is a meaningful thing to
report even without real footage. See ``docs/evaluation.md``.

    python -m retail_video_analytics.evaluate --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import VideoAnalyticsPipeline
from .synthetic import SimulatedDetector, default_actor_paths, generate_synthetic_detections

SCENARIOS = [
    {"name": "clean", "noise_std": 0.0, "miss_rate": 0.0},
    {"name": "moderate_noise", "noise_std": 3.0, "miss_rate": 0.05},
    {"name": "high_noise", "noise_std": 8.0, "miss_rate": 0.2},
]


def run_evaluation(config_path: str, num_frames: int | None = None) -> list[dict]:
    config = load_config(config_path)
    if num_frames is not None:
        config.synthetic_num_frames = num_frames
    num_frames = config.synthetic_num_frames
    paths = default_actor_paths(640, 480, num_frames=num_frames)
    ground_truth = generate_synthetic_detections(paths, num_frames=num_frames)
    true_actor_count = len(paths)

    results = []
    for scenario in SCENARIOS:
        detector = SimulatedDetector(
            ground_truth,
            noise_std=scenario["noise_std"],
            miss_rate=scenario["miss_rate"],
            seed=0,
        )
        pipeline = VideoAnalyticsPipeline(config, detector=detector)
        summary = pipeline.run(persist=False)
        counting_error = abs(summary.unique_customers - true_actor_count)
        results.append(
            {
                "scenario": scenario["name"],
                "noise_std": scenario["noise_std"],
                "miss_rate": scenario["miss_rate"],
                "true_actor_count": true_actor_count,
                "counted_unique_customers": summary.unique_customers,
                "counting_error": counting_error,
                "max_concurrent_customers": summary.max_concurrent_customers,
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate counting accuracy vs synthetic ground truth")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args(argv)

    results = run_evaluation(args.config, num_frames=args.frames)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
