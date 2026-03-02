import numpy as np


def sharpe_ratio(returns: np.ndarray, annual_factor: float = 252.0) -> float:
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(annual_factor))


def sortino_ratio(returns: np.ndarray, annual_factor: float = 252.0) -> float:
    downside = returns[returns < 0]
    down_std = downside.std() if len(downside) > 0 else 0.0
    if down_std == 0:
        return 0.0
    return float(returns.mean() / down_std * np.sqrt(annual_factor))


def max_drawdown(returns: np.ndarray) -> float:
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(dd.max()) if len(dd) > 0 else 0.0


def calmar_ratio(returns: np.ndarray, annual_factor: float = 252.0) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return annual_return(returns, annual_factor) / mdd


def annual_return(returns: np.ndarray, annual_factor: float = 252.0) -> float:
    total = np.prod(1.0 + returns)
    n = len(returns)
    if n == 0:
        return 0.0
    return float(total ** (annual_factor / n) - 1.0)


def annual_volatility(returns: np.ndarray, annual_factor: float = 252.0) -> float:
    return float(returns.std() * np.sqrt(annual_factor))


def turnover(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.abs(np.diff(positions)).mean())


def cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional VaR (expected shortfall) of returns.

    Returns the mean of the worst alpha-fraction outcomes.
    """
    if len(returns) == 0:
        return 0.0
    a = float(np.clip(alpha, 1e-6, 1.0))
    n_tail = max(1, int(np.ceil(a * len(returns))))
    tail = np.partition(np.asarray(returns, dtype=float), n_tail - 1)[:n_tail]
    return float(np.mean(tail))


def expected_log_growth(returns: np.ndarray, annual_factor: float = 252.0) -> float:
    """Annualized expected log-growth rate."""
    if len(returns) == 0:
        return 0.0
    r = np.asarray(returns, dtype=float)
    r = np.clip(r, -0.999999, None)
    return float(np.mean(np.log1p(r)) * annual_factor)


def tail_hit_rate(returns: np.ndarray, sigma: float = 2.0) -> float:
    """Rate of returns breaching mean - sigma * std."""
    if len(returns) == 0:
        return 0.0
    r = np.asarray(returns, dtype=float)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    threshold = mu - sigma * sd
    return float(np.mean(r < threshold))
