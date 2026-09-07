"""Generate a traffic/dwell heatmap from a stored pipeline run.

    python -m retail_video_analytics.heatmap_cli --db data/analytics.db --run-id 1

Reads the run's persisted position samples and frame geometry from SQLite,
renders the density grid, and writes ``heatmap_<run_id>.png`` +
``heatmap_<run_id>.json`` into the output directory.
"""

from __future__ import annotations

import argparse
import sys

from .analytics.heatmap import render_heatmap, save_heatmap
from .storage.db import AnalyticsStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a heatmap from a stored analytics run"
    )
    parser.add_argument("--db", default="data/analytics.db", help="SQLite database path")
    parser.add_argument("--run-id", type=int, required=True, help="run id to render")
    parser.add_argument(
        "--out", default="data/heatmaps", help="directory for the PNG + JSON output"
    )
    parser.add_argument("--cell-size", type=int, default=20, help="grid cell size in pixels")
    args = parser.parse_args(argv)

    with AnalyticsStore(args.db) as store:
        try:
            run = store.get_run(args.run_id)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        samples = store.list_position_samples(args.run_id)
        if not samples:
            print(
                f"error: run {args.run_id} has no stored position samples; "
                "re-run the pipeline with persistence enabled",
                file=sys.stderr,
            )
            return 1

        width = run["frame_width"] or 640
        height = run["frame_height"] or 480
        grid = render_heatmap(samples, width, height, cell_size=args.cell_size)
        artifacts = save_heatmap(grid, args.out, args.run_id)

    print(f"wrote {artifacts.png_path}")
    print(f"wrote {artifacts.data_path}")
    print(f"samples: {grid.sample_count}, grid: {grid.counts.shape[0]}x{grid.counts.shape[1]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
