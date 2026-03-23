from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonitorConfig:
    rolling_window: int = 63
    sharpe_threshold: float = 0.0
    drawdown_threshold: float = 0.10
    min_observations: int = 20


@dataclass
class MonitorStatus:
    alpha_id: str
    rolling_sharpe: float
    rolling_log_growth: float
    rolling_max_dd: float
    is_degraded: bool
    degradation_reasons: list[str]

    _ROLLING_FITNESS_MAP = {"sharpe": "rolling_sharpe", "log_growth": "rolling_log_growth"}

    def rolling_fitness(self, metric: str = "sharpe") -> float:
        return getattr(self, self._ROLLING_FITNESS_MAP[metric])


class AlphaMonitor:
    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()
        self._returns: dict[str, list[float]] = {}

    def record(self, alpha_id: str, daily_return: float) -> None:
        if alpha_id not in self._returns:
            self._returns[alpha_id] = []
        self._returns[alpha_id].append(daily_return)

    def record_batch(self, alpha_id: str, returns: list[float] | np.ndarray) -> None:
        if alpha_id not in self._returns:
            self._returns[alpha_id] = []
        self._returns[alpha_id].extend(float(r) for r in returns)

    def check(self, alpha_id: str) -> MonitorStatus:
        cfg = self.config
        rets = self._returns.get(alpha_id, [])

        if len(rets) < cfg.min_observations:
            return MonitorStatus(
                alpha_id=alpha_id,
                rolling_sharpe=0.0,
                rolling_log_growth=0.0,
                rolling_max_dd=0.0,
                is_degraded=False,
                degradation_reasons=[],
            )

        recent = np.array(rets[-cfg.rolling_window:])
        std = recent.std()
        rolling_sharpe = float(recent.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        r_clipped = np.clip(recent, -0.999999, None)
        rolling_log_growth = float(np.mean(np.log1p(r_clipped)) * 252)

        cum = np.cumprod(1 + recent)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / peak
        rolling_max_dd = float(dd.max())

        reasons: list[str] = []
        if rolling_sharpe < cfg.sharpe_threshold:
            reasons.append(f"Rolling Sharpe {rolling_sharpe:.3f} < {cfg.sharpe_threshold}")
        if rolling_max_dd > cfg.drawdown_threshold:
            reasons.append(f"Rolling MaxDD {rolling_max_dd:.1%} > {cfg.drawdown_threshold:.1%}")

        return MonitorStatus(
            alpha_id=alpha_id,
            rolling_sharpe=rolling_sharpe,
            rolling_log_growth=rolling_log_growth,
            rolling_max_dd=rolling_max_dd,
            is_degraded=len(reasons) > 0,
            degradation_reasons=reasons,
        )

    def check_all(self) -> list[MonitorStatus]:
        return [self.check(alpha_id) for alpha_id in self._returns]

    def clear(self, alpha_id: str) -> None:
        self._returns.pop(alpha_id, None)


@dataclass
class RegimeStatus:
    current_vol_regime: str
    vol_ratio: float
    trend_regime: str
    drift_score: float


class RegimeDetector:
    def __init__(self, short_window: int = 21, long_window: int = 63):
        self._short = short_window
        self._long = long_window

    def detect(self, returns: np.ndarray) -> RegimeStatus:
        if len(returns) < self._long:
            return RegimeStatus("normal", 1.0, "neutral", 0.0)

        short_vol = float(np.std(returns[-self._short:]))
        long_vol = float(np.std(returns[-self._long:]))
        vol_ratio = short_vol / long_vol if long_vol > 1e-8 else 1.0

        if vol_ratio > 1.5:
            vol_regime = "high"
        elif vol_ratio < 0.7:
            vol_regime = "low"
        else:
            vol_regime = "normal"

        r = returns[-self._long:]
        autocorr = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        if np.isnan(autocorr):
            autocorr = 0.0
        if autocorr > 0.1:
            trend = "trending"
        elif autocorr < -0.1:
            trend = "mean_reverting"
        else:
            trend = "neutral"

        from scipy.stats import ks_2samp

        half = self._long // 2
        first = returns[-self._long:-half]
        second = returns[-half:]
        stat, _ = ks_2samp(first, second)

        return RegimeStatus(
            current_vol_regime=vol_regime,
            vol_ratio=vol_ratio,
            trend_regime=trend,
            drift_score=float(stat),
        )
