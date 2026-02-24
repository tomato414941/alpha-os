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
    pbo_max: float = 0.50
    dsr_pvalue_max: float = 0.05
    fdr_pass_required: bool = True
    max_correlation: float = 0.5
    min_n_days: int = 200


def adoption_gate(
    oos_sharpe: float,
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
