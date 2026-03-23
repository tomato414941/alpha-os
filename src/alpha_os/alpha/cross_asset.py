"""Compatibility wrapper for cross-asset research helpers."""

from alpha_os.research.cross_asset import (
    DEFAULT_HORIZONS,
    CrossAssetResult,
    evaluate_cross_asset,
    evaluate_cross_asset_multi_horizon,
    forward_returns as _forward_returns,
    mean_cross_asset_fitness,
    residualize_forward_returns as _residualize,
)

__all__ = [
    "DEFAULT_HORIZONS",
    "CrossAssetResult",
    "_forward_returns",
    "_residualize",
    "evaluate_cross_asset",
    "evaluate_cross_asset_multi_horizon",
    "mean_cross_asset_fitness",
]
