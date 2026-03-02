"""Governance gates — alpha adoption criteria checklist."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    passed: bool
    checks: dict[str, bool]
    reasons: list[str]


@dataclass
class GateConfig:
    oos_sharpe_min: float = 0.5
    oos_log_growth_min: float = -1.0
    oos_cvar_abs_max: float = 1.0
    oos_tail_hit_rate_max: float = 1.0
    pbo_max: float = 1.0
    dsr_pvalue_max: float = 1.0   # disabled — lifecycle manages quality
    fdr_pass_required: bool = False
    max_correlation: float = 0.5
    min_n_days: int = 200


def adoption_gate(
    oos_sharpe: float,
    oos_log_growth: float,
    oos_cvar_95: float,
    oos_tail_hit_rate: float,
    pbo: float,
    dsr_pvalue: float,
    fdr_passed: bool,
    avg_correlation: float,
    n_days: int,
    config: GateConfig | None = None,
) -> GateResult:
    """Evaluate whether an alpha meets adoption criteria.

    All checks must pass for adoption.
    """
    cfg = config or GateConfig()
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    checks["oos_sharpe"] = oos_sharpe >= cfg.oos_sharpe_min
    if not checks["oos_sharpe"]:
        reasons.append(f"OOS Sharpe {oos_sharpe:.3f} < {cfg.oos_sharpe_min}")

    checks["oos_log_growth"] = oos_log_growth >= cfg.oos_log_growth_min
    if not checks["oos_log_growth"]:
        reasons.append(
            f"OOS log-growth {oos_log_growth:.3f} < {cfg.oos_log_growth_min}"
        )

    cvar_abs = abs(min(float(oos_cvar_95), 0.0))
    checks["oos_cvar"] = cvar_abs <= cfg.oos_cvar_abs_max
    if not checks["oos_cvar"]:
        reasons.append(f"OOS |CVaR95| {cvar_abs:.3%} > {cfg.oos_cvar_abs_max:.3%}")

    checks["oos_tail_hit_rate"] = oos_tail_hit_rate <= cfg.oos_tail_hit_rate_max
    if not checks["oos_tail_hit_rate"]:
        reasons.append(
            f"OOS tail-hit {oos_tail_hit_rate:.3%} > {cfg.oos_tail_hit_rate_max:.3%}"
        )

    checks["pbo"] = pbo <= cfg.pbo_max
    if not checks["pbo"]:
        reasons.append(f"PBO {pbo:.3f} > {cfg.pbo_max}")

    checks["dsr"] = dsr_pvalue <= cfg.dsr_pvalue_max
    if not checks["dsr"]:
        reasons.append(f"DSR p-value {dsr_pvalue:.3f} > {cfg.dsr_pvalue_max}")

    checks["fdr"] = fdr_passed or not cfg.fdr_pass_required
    if not checks["fdr"]:
        reasons.append("Failed FDR correction")

    checks["correlation"] = avg_correlation <= cfg.max_correlation
    if not checks["correlation"]:
        reasons.append(f"Avg correlation {avg_correlation:.3f} > {cfg.max_correlation}")

    checks["n_days"] = n_days >= cfg.min_n_days
    if not checks["n_days"]:
        reasons.append(f"Only {n_days} days < {cfg.min_n_days} minimum")

    return GateResult(
        passed=all(checks.values()),
        checks=checks,
        reasons=reasons,
    )
