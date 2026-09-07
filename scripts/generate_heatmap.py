"""Thin wrapper so the heatmap generator runs without installing the package.

    python scripts/generate_heatmap.py --db data/analytics.db --run-id 1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_video_analytics.heatmap_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
