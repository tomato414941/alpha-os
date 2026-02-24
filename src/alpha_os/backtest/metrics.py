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
