"""Compatibility wrapper for legacy funnel summaries."""

from alpha_os.config import asset_data_dir
from alpha_os.evolution.discovery_pool import DiscoveryPool
from alpha_os.legacy import funnel as _legacy
from alpha_os.legacy.funnel import (
    FunnelSummary,
    SourceFunnelSummary,
)


def load_funnel_summary(asset: str) -> FunnelSummary:
    _legacy.asset_data_dir = asset_data_dir
    _legacy.DiscoveryPool = DiscoveryPool
    return _legacy.load_funnel_summary(asset)

__all__ = [
    "FunnelSummary",
    "SourceFunnelSummary",
    "load_funnel_summary",
]
