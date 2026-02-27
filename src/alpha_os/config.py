"""Configuration loading from TOML files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"


@dataclass
class APIConfig:
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    retry_count: int = 2
    retry_backoff: float = 1.0


@dataclass
class GenerationConfig:
    candidates_per_cycle: int = 10000
    max_depth: int = 3
    windows: list[int] = field(default_factory=lambda: [5, 10, 20, 30, 60])


@dataclass
class BacktestConfig:
    commission_pct: float = 0.10
    slippage_pct: float = 0.05
    min_days: int = 200
    eval_window_days: int = 0  # 0 = all data; >0 = trailing N days


@dataclass
class ValidationConfig:
    oos_sharpe_min: float = 0.5
    pbo_max: float = 1.0
    dsr_pvalue_max: float = 1.0
    fdr_threshold: float = 0.05
    n_cv_folds: int = 5
    embargo_days: int = 5


@dataclass
class RiskConfig:
    target_vol_pct: float = 15.0
    dd_stage1_pct: float = 5.0
    dd_stage2_pct: float = 10.0
    dd_stage3_pct: float = 15.0

    def to_manager_config(self):
        """Convert user-facing percentages to decimal RiskManagerConfig."""
        from alpha_os.risk.manager import RiskManagerConfig
        return RiskManagerConfig(
            target_vol=self.target_vol_pct / 100.0,
            dd_stage1_pct=self.dd_stage1_pct / 100.0,
            dd_stage2_pct=self.dd_stage2_pct / 100.0,
            dd_stage3_pct=self.dd_stage3_pct / 100.0,
        )


@dataclass
class TradingConfig:
    initial_capital: float = 10000.0


@dataclass
class PaperTradingConfig:
    max_position_pct: float = 1.0
    min_trade_usd: float = 10.0


@dataclass
class ForwardTestConfig:
    check_interval: int = 14400
    min_forward_days: int = 30
    degradation_window: int = 63


@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    paper: PaperTradingConfig = field(default_factory=PaperTradingConfig)
    forward: ForwardTestConfig = field(default_factory=ForwardTestConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        if path is None:
            path = CONFIG_DIR / "default.toml"
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(
            api=APIConfig(**raw.get("api", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            validation=ValidationConfig(**raw.get("validation", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            trading=TradingConfig(**raw.get("trading", {})),
            paper=PaperTradingConfig(**raw.get("paper", {})),
            forward=ForwardTestConfig(**raw.get("forward", {})),
        )
