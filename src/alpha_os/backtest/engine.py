from dataclasses import dataclass

import numpy as np

from .cost_model import CostModel
from . import metrics


@dataclass
class BacktestResult:
    alpha_id: str
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    annual_return: float
    annual_vol: float
    turnover: float
    cvar_95: float
    expected_log_growth: float
    tail_hit_rate: float
    n_days: int


def _normalize_positions(signal: np.ndarray) -> np.ndarray:
    s = signal.copy().astype(float)
    s[np.isnan(s)] = 0.0
    std = s.std()
    if std == 0:
        return np.clip(np.sign(s), -1.0, 1.0)
    return np.clip(s / std, -1.0, 1.0)


class BacktestEngine:
    def __init__(self, cost_model: CostModel | None = None):
        self._cost = cost_model or CostModel()

    def run(self, alpha_signal: np.ndarray, prices: np.ndarray,
            alpha_id: str = "") -> BacktestResult:
        pos = _normalize_positions(alpha_signal)
        rets = np.diff(prices) / prices[:-1]
        n = min(len(pos) - 1, len(rets))
        strat_rets = pos[:n] * rets[:n]
        cost = np.abs(np.diff(pos[:n + 1])) * self._cost.one_way_cost
        net = strat_rets - cost[:n]
        return BacktestResult(
            alpha_id=alpha_id,
            sharpe=metrics.sharpe_ratio(net),
            sortino=metrics.sortino_ratio(net),
            max_drawdown=metrics.max_drawdown(net),
            calmar=metrics.calmar_ratio(net),
            annual_return=metrics.annual_return(net),
            annual_vol=metrics.annual_volatility(net),
            turnover=metrics.turnover(pos),
            cvar_95=metrics.cvar(net, alpha=0.05),
            expected_log_growth=metrics.expected_log_growth(net),
            tail_hit_rate=metrics.tail_hit_rate(net, sigma=2.0),
            n_days=len(net),
        )

    def run_batch(self, signals: np.ndarray, prices: np.ndarray,
                  alpha_ids: list[str] | None = None) -> list[BacktestResult]:
        n_alphas = signals.shape[0]
        ids = alpha_ids or [f"alpha_{i}" for i in range(n_alphas)]

        sigs = signals.astype(float).copy()
        sigs[np.isnan(sigs)] = 0.0
        stds = sigs.std(axis=1, keepdims=True)
        zero_std_mask = stds.flatten() == 0
        stds[stds == 0] = 1.0
        pos = np.clip(sigs / stds, -1.0, 1.0)
        pos[zero_std_mask] = np.clip(np.sign(sigs[zero_std_mask]), -1.0, 1.0)

        rets = np.diff(prices) / prices[:-1]
        n = min(pos.shape[1] - 1, len(rets))
        strat_rets = pos[:, :n] * rets[:n]
        dpos = np.abs(np.diff(pos[:, :n + 1], axis=1))
        cost = dpos[:, :n] * self._cost.one_way_cost
        net = strat_rets - cost

        results = []
        for i in range(n_alphas):
            r = net[i]
            p = pos[i]
            results.append(BacktestResult(
                alpha_id=ids[i],
                sharpe=metrics.sharpe_ratio(r),
                sortino=metrics.sortino_ratio(r),
                max_drawdown=metrics.max_drawdown(r),
                calmar=metrics.calmar_ratio(r),
                annual_return=metrics.annual_return(r),
                annual_vol=metrics.annual_volatility(r),
                turnover=metrics.turnover(p),
                cvar_95=metrics.cvar(r, alpha=0.05),
                expected_log_growth=metrics.expected_log_growth(r),
                tail_hit_rate=metrics.tail_hit_rate(r, sigma=2.0),
                n_days=len(r),
            ))
        return results
