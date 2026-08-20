"""Retail Video Analytics Platform.

A from-scratch, CPU-only reference implementation of a retail video
analytics pipeline: detection -> tracking -> zone/dwell-time logic ->
storage. See docs/architecture.md for the full design.
"""

from .config import PipelineConfig, load_config
from .pipeline import RunSummary, VideoAnalyticsPipeline

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "load_config",
    "RunSummary",
    "VideoAnalyticsPipeline",
    "__version__",
]
