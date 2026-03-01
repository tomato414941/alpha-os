"""TacticalTrader — Layer 2 hourly alpha evaluation.

Evaluates hourly derivative signals and produces a tactical score that
modulates the strategic (Layer 3 daily) bias. When no L2 alphas exist,
returns tactical_score=0 and combined=strategic_bias (transparent pass-through).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np

from ..alpha.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..alpha.registry import AlphaRegistry, AlphaState
from ..alpha.combiner import (
    WeightedCombinerConfig,
    compute_weights,
    weighted_combine_scalar,
)
from ..alpha.monitor import AlphaMonitor, MonitorConfig
from ..config import Config, asset_data_dir
from ..data.store import DataStore
from ..data.universe import build_hourly_feature_list
from ..dsl import parse

logger = logging.getLogger(__name__)


@dataclass
class TacticalSignal:
    timestamp: str
    tactical_score: float       # -1 to 1
    strategic_bias: float       # from Layer 3
    combined_signal: float      # agreement-modulated
    n_alphas_evaluated: int
    confidence: float


class TacticalTrader:
    """Layer 2 hourly tactical alpha evaluator.

    Uses a separate registry and data cache (alpha_registry_l2.db,
    alpha_cache_l2.db) to avoid conflicts with the daily Layer 3 system.
    """

    def __init__(
        self,
        asset: str,
        config: Config,
        registry: AlphaRegistry | None = None,
        store: DataStore | None = None,
    ):
        self.asset = asset
        self.config = config
        self.resolution = "1h"
        self.features = build_hourly_feature_list(asset)

        adir = asset_data_dir(asset)
        self.registry = registry or AlphaRegistry(
            db_path=adir / "alpha_registry_l2.db",
        )

        if store is not None:
            self.store = store
        else:
            from signal_noise.client import SignalClient
            client = SignalClient(
                base_url=config.api.base_url,
                timeout=config.api.timeout,
            )
            from ..config import DATA_DIR
            self.store = DataStore(DATA_DIR / "alpha_cache_l2.db", client)

        mon_cfg = MonitorConfig(rolling_window=config.forward.degradation_window)
        self.monitor = AlphaMonitor(config=mon_cfg)
        self._wcfg = WeightedCombinerConfig()
        self._diversity_cache: dict[str, float] = {}

    def run_cycle(self, strategic_bias: float = 0.0) -> TacticalSignal:
        """Evaluate L2 alphas and modulate strategic bias.

        Agreement logic:
        - Same direction → amplify: combined = bias * (1 + |tactical| * 0.5)
        - Opposite direction → attenuate: combined = bias * (1 - |tactical| * 0.5)
        - No L2 alphas → combined = strategic_bias (transparent pass-through)
        """
        now_str = date.today().isoformat()

        # 1. Sync hourly data
        try:
            self.store.sync(self.features, resolution=self.resolution)
        except Exception:
            logger.warning("L2 API sync failed — using cached data")

        # 2. Get active L2 alphas
        active = self.registry.list_by_state(AlphaState.ACTIVE)
        probation = self.registry.list_by_state(AlphaState.PROBATION)
        all_alphas = active + probation

        if not all_alphas:
            return TacticalSignal(
                timestamp=now_str,
                tactical_score=0.0,
                strategic_bias=strategic_bias,
                combined_signal=strategic_bias,
                n_alphas_evaluated=0,
                confidence=0.0,
            )

        # 3. Build hourly data matrix
        matrix = self.store.get_matrix(self.features, resolution=self.resolution)
        if len(matrix) < 2:
            return TacticalSignal(
                timestamp=now_str,
                tactical_score=0.0,
                strategic_bias=strategic_bias,
                combined_signal=strategic_bias,
                n_alphas_evaluated=0,
                confidence=0.0,
            )

        data = {col: matrix[col].values for col in matrix.columns}
        n_rows = len(matrix)

        # 4. Evaluate each alpha
        alpha_signals: dict[str, float] = {}
        n_evaluated = 0
        for record in all_alphas:
            try:
                expr = parse(record.expression)
                signal = evaluate_expression(expr, data, n_rows)
                signal_norm = normalize_signal(signal)
                alpha_signals[record.alpha_id] = float(signal_norm[-1])
                n_evaluated += 1
            except EvaluationError:
                continue

        if not alpha_signals:
            return TacticalSignal(
                timestamp=now_str,
                tactical_score=0.0,
                strategic_bias=strategic_bias,
                combined_signal=strategic_bias,
                n_alphas_evaluated=0,
                confidence=0.0,
            )

        # 5. Combine with quality × diversity
        alpha_ids = list(alpha_signals.keys())
        sharpes_list = []
        diversity_list = []
        for aid in alpha_ids:
            status = self.monitor.check(aid)
            sharpes_list.append(max(status.rolling_sharpe, 0.0))
            diversity_list.append(self._diversity_cache.get(aid, 1.0))

        sharpes_np = np.array(sharpes_list)
        diversity_np = np.array(diversity_list)
        w = compute_weights(sharpes_np, diversity_np, min_weight=self._wcfg.min_weight)
        weights_dict = {aid: float(w[i]) for i, aid in enumerate(alpha_ids)}
        tactical_score = weighted_combine_scalar(alpha_signals, weights_dict)

        # 6. Agreement modulation
        combined = self._modulate(strategic_bias, tactical_score)

        confidence = abs(tactical_score) * min(n_evaluated / 5.0, 1.0)

        return TacticalSignal(
            timestamp=now_str,
            tactical_score=tactical_score,
            strategic_bias=strategic_bias,
            combined_signal=combined,
            n_alphas_evaluated=n_evaluated,
            confidence=confidence,
        )

    @staticmethod
    def _modulate(bias: float, tactical: float) -> float:
        """Agreement-based modulation of strategic bias by tactical score."""
        if abs(tactical) < 1e-8:
            return bias

        same_direction = (bias * tactical) > 0
        if same_direction:
            combined = bias * (1 + abs(tactical) * 0.5)
        else:
            combined = bias * (1 - abs(tactical) * 0.5)

        return float(np.clip(combined, -1.0, 1.0))

    def close(self) -> None:
        self.store.close()
        self.registry.close()
