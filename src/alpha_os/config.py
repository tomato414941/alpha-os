"""Configuration loading from TOML files."""

from __future__ import annotations

import logging
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_DIR / "config"
DATA_DIR = PROJECT_DIR / "data"

_logger = logging.getLogger(__name__)

_BTC_MIGRATED = False


def asset_data_dir(asset: str) -> Path:
    """Return per-asset data directory, creating it if needed."""
    d = DATA_DIR / asset.upper()
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics").mkdir(exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    if asset.upper() == "BTC":
        _maybe_migrate_btc(d)
    return d


def _maybe_migrate_btc(adir: Path) -> None:
    """One-time migration: move flat data/ files into data/BTC/."""
    global _BTC_MIGRATED
    if _BTC_MIGRATED:
        return
    _BTC_MIGRATED = True

    flat_files = [
        "alpha_registry.db",
        "forward_returns.db",
        "paper_trading.db",
        "audit.jsonl",
    ]
    metric_files = [
        "metrics/circuit_breaker.json",
        "metrics/testnet_validation.json",
        "metrics/testnet_reports.jsonl",
    ]

    for f in flat_files:
        src = DATA_DIR / f
        dst = adir / f
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            _logger.info("Migrated %s → %s", src, dst)

    for f in metric_files:
        src = DATA_DIR / f
        dst = adir / f
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            _logger.info("Migrated %s → %s", src, dst)


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
    bloat_penalty: float = 0.01
    depth_penalty: float = 0.0
    similarity_penalty: float = 0.0


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
class TestnetConfig:
    target_success_days: int = 10
    max_acceptable_slippage_bps: float = 50.0


@dataclass
class EventDrivenConfig:
    min_interval: float = 900.0
    max_interval: float = 14400.0
    subscribe_pattern: str = "funding_rate_*,liq_*"
    anomaly_trigger: bool = True


@dataclass
class LifecycleTomlConfig:
    oos_sharpe_min: float = 0.05
    probation_sharpe_min: float = 0.0
    dormant_sharpe_max: float = -0.5
    dormant_revival_sharpe: float = 0.0
    correlation_max: float = 0.5


@dataclass
class ExecutionTomlConfig:
    imbalance_threshold: float = 0.1
    vpin_threshold: float = 0.5
    spread_threshold_bps: float = 5.0
    max_slices: int = 5


@dataclass
class DistributionalConfig:
    enabled: bool = False
    window: int = 63
    min_samples: int = 20
    tail_sigma: float = 2.0
    cvar_alpha: float = 0.05
    max_left_tail_prob: float = 0.10
    max_cvar_abs: float = 0.03
    kelly_fraction: float = 0.50
    max_kelly_leverage: float = 1.0


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
    testnet: TestnetConfig = field(default_factory=TestnetConfig)
    event_driven: EventDrivenConfig = field(default_factory=EventDrivenConfig)
    lifecycle: LifecycleTomlConfig = field(default_factory=LifecycleTomlConfig)
    execution: ExecutionTomlConfig = field(default_factory=ExecutionTomlConfig)
    distributional: DistributionalConfig = field(default_factory=DistributionalConfig)

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
            testnet=TestnetConfig(**raw.get("testnet", {})),
            event_driven=EventDrivenConfig(**raw.get("event_driven", {})),
            lifecycle=LifecycleTomlConfig(**raw.get("lifecycle", {})),
            execution=ExecutionTomlConfig(**raw.get("execution", {})),
            distributional=DistributionalConfig(**raw.get("distributional", {})),
        )
