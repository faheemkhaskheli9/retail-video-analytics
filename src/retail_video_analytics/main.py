"""CLI entry point.

    python -m retail_video_analytics.main --config configs/default.yaml
    python -m retail_video_analytics.main --config configs/default.yaml --video path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .pipeline import VideoAnalyticsPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retail video analytics pipeline runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to a pipeline YAML config.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Override the config's video_source (path to a video file, or 'synthetic').",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing results to the configured database.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.video:
        config.video_source = args.video

    pipeline = VideoAnalyticsPipeline(config)
    summary = pipeline.run(persist=not args.no_persist)

    print(json.dumps(summary.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
