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

TRADING_MODE_SPOT_LONG_ONLY = "spot_long_only"
TRADING_MODE_FUTURES_LONG_SHORT = "futures_long_short"
_TRADING_MODES = {
    TRADING_MODE_SPOT_LONG_ONLY,
    TRADING_MODE_FUTURES_LONG_SHORT,
}


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
    ]
    renamed_metric_files = {
        "metrics/testnet_validation.json": "metrics/testnet_readiness.json",
        "metrics/testnet_reports.jsonl": "metrics/testnet_readiness_reports.jsonl",
    }

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

    for old_name, new_name in renamed_metric_files.items():
        flat_src = DATA_DIR / old_name
        asset_legacy = adir / old_name
        dst = adir / new_name
        if flat_src.exists() and not dst.exists():
            shutil.move(str(flat_src), str(dst))
            _logger.info("Migrated %s → %s", flat_src, dst)
        if asset_legacy.exists() and not dst.exists():
            shutil.move(str(asset_legacy), str(dst))
            _logger.info("Renamed %s → %s", asset_legacy, dst)


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
    max_vol_scale: float = 1.5

    def to_manager_config(self):
        """Convert user-facing percentages to decimal RiskManagerConfig."""
        from alpha_os.risk.manager import RiskManagerConfig
        return RiskManagerConfig(
            target_vol=self.target_vol_pct / 100.0,
            dd_stage1_pct=self.dd_stage1_pct / 100.0,
            dd_stage2_pct=self.dd_stage2_pct / 100.0,
            dd_stage3_pct=self.dd_stage3_pct / 100.0,
            max_vol_scale=self.max_vol_scale,
        )


@dataclass
class TradingConfig:
    initial_capital: float = 10000.0
    mode: str = TRADING_MODE_SPOT_LONG_ONLY

    @property
    def supports_short(self) -> bool:
        if self.mode == TRADING_MODE_SPOT_LONG_ONLY:
            return False
        if self.mode == TRADING_MODE_FUTURES_LONG_SHORT:
            return True
        raise ValueError(
            f"Unknown trading mode: {self.mode!r}. Expected one of {sorted(_TRADING_MODES)}"
        )


@dataclass
class PaperTradingConfig:
    max_position_pct: float = 1.0
    min_trade_usd: float = 10.0
    rebalance_deadband_usd: float = 0.0
    max_trading_alphas: int = 30
    combine_mode: str = "consensus"  # "consensus" | "voting" | "map_elites"


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
class LiveQualityConfig:
    min_observations: int = 20
    full_weight_observations: int = 63
    shortlist_preselect_factor: int = 20
    dormant_revival_min_observations: int = 20
    weight_confidence_floor: float = 1.0
    weight_confidence_power: float = 1.0


@dataclass
class DeploymentConfig:
    max_alphas: int = 150
    max_replacements: int = 10
    promotion_margin: float = 0.05
    signal_similarity_max: float = 0.995
    signal_similarity_lookback: int = 252
    max_feature_occurrences: int = 0


@dataclass
class LifecycleTomlConfig:
    candidate_quality_min: float = 0.05
    active_quality_min: float = 0.0
    dormant_revival_quality: float = 0.0
    correlation_max: float = 0.5

    @property
    def oos_quality_min(self) -> float:
        return self.candidate_quality_min

    @oos_quality_min.setter
    def oos_quality_min(self, value: float) -> None:
        self.candidate_quality_min = value

    @property
    def probation_quality_min(self) -> float:
        return self.active_quality_min

    @probation_quality_min.setter
    def probation_quality_min(self, value: float) -> None:
        self.active_quality_min = value

    @property
    def dormant_quality_max(self) -> float:
        return self.active_quality_min

    @dormant_quality_max.setter
    def dormant_quality_max(self, value: float) -> None:
        self.active_quality_min = value


@dataclass
class ExecutionTomlConfig:
    imbalance_threshold: float = 0.3
    vpin_threshold: float = 0.85
    spread_threshold_bps: float = 5.0
    max_slices: int = 5
    signal_lookback_minutes: int = 15
    max_signal_age_seconds: int = 300
    commission_pct: float = 0.10
    modeled_slippage_pct: float = 0.05

    def to_cost_model(self):
        """Build the shared runtime cost model."""
        from alpha_os.execution.costs import ExecutionCostModel

        return ExecutionCostModel(
            commission_pct=self.commission_pct,
            modeled_slippage_pct=self.modeled_slippage_pct,
        )


@dataclass
class GateTomlConfig:
    oos_log_growth_min: float = 0.0
    oos_cvar_abs_max: float = 0.05
    oos_tail_hit_rate_max: float = 0.10


@dataclass
class RegimeConfig:
    enabled: bool = True
    short_window: int = 21
    long_window: int = 63
    drift_threshold: float = 0.3
    drift_position_scale_min: float = 0.5


@dataclass
class StabilityConfig:
    """Path A: alpha stabilization settings."""
    tenure_bonus_max: float = 0.2       # max quality bonus for long-lived alphas
    tenure_half_life_days: float = 7.0  # days to reach 50% of max bonus
    top_n_protect: int = 10             # top-N alphas get demotion cooldown
    demotion_cooldown_days: float = 3.0 # min days before top-N can be demoted


@dataclass
class EvoDaemonConfig:
    enabled: bool = False
    pop_size: int = 80
    n_generations: int = 15
    round_interval: int = 300
    memory_limit_mb: int = 400
    batch_size: int = 500
    evo_mode: str = "legacy"  # "legacy" | "map_elites"
    feature_subset_k: int = 27  # √753 ≈ 27, features per subset


@dataclass
class AdmissionConfig:
    enabled: bool = False
    poll_interval: int = 1800
    batch_size: int = 100
    min_queue_size: int = 10
    max_active_alphas: int = 0
    diversity_recompute_days: int = 63
    incremental_diversity: bool = True


@dataclass
class LifecycleDaemonConfig:
    enabled: bool = False


@dataclass
class Config:
    fitness_metric: str = "sharpe"
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
    live_quality: LiveQualityConfig = field(default_factory=LiveQualityConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    lifecycle: LifecycleTomlConfig = field(default_factory=LifecycleTomlConfig)
    execution: ExecutionTomlConfig = field(default_factory=ExecutionTomlConfig)
    gate: GateTomlConfig = field(default_factory=GateTomlConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    evo_daemon: EvoDaemonConfig = field(default_factory=EvoDaemonConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    lifecycle_daemon: LifecycleDaemonConfig = field(default_factory=LifecycleDaemonConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)

    def to_monitor_config(self):
        """Build the monitor config used by live forward-quality checks."""
        from alpha_os.alpha.monitor import MonitorConfig

        return MonitorConfig(
            rolling_window=self.forward.degradation_window,
            min_observations=self.live_quality.min_observations,
        )

    def to_lifecycle_config(self):
        """Build the lifecycle config used by live state transitions."""
        from alpha_os.alpha.lifecycle import LifecycleConfig

        return LifecycleConfig(
            candidate_quality_min=self.lifecycle.candidate_quality_min,
            active_quality_min=self.lifecycle.active_quality_min,
            pbo_max=self.validation.pbo_max,
            dsr_pvalue_max=self.validation.dsr_pvalue_max,
            correlation_max=self.lifecycle.correlation_max,
            dormant_revival_quality=self.lifecycle.dormant_revival_quality,
        )

    def estimate_alpha_quality(self, prior_quality: float, returns):
        """Blend historical and live quality using the current runtime config."""
        from alpha_os.alpha.quality import blend_quality

        return blend_quality(
            prior_quality,
            returns,
            metric=self.fitness_metric,
            rolling_window=self.forward.degradation_window,
            min_observations=self.live_quality.min_observations,
            full_weight_observations=self.live_quality.full_weight_observations,
        )

    @staticmethod
    def _filter(dc_class: type, raw: dict) -> dict:
        """Filter dict to only keys that dc_class accepts."""
        valid = {f.name for f in dc_class.__dataclass_fields__.values()}
        return {k: v for k, v in raw.items() if k in valid}

    @staticmethod
    def _read_toml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "rb") as f:
            return tomllib.load(f)

    @staticmethod
    def _merge_dicts(base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = Config._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        default_path = CONFIG_DIR / "default.toml"
        if path is None:
            raw = cls._read_toml(default_path)
        else:
            path = Path(path)
            if path.resolve() == default_path.resolve():
                raw = cls._read_toml(path)
            else:
                raw = cls._merge_dicts(
                    cls._read_toml(default_path),
                    cls._read_toml(path),
                )
        if not raw:
            return cls()
        # Backward compat: map old lifecycle sharpe keys to quality keys
        lc_raw = dict(raw.get("lifecycle", {}))
        _LC_ALIASES = {
            "candidate_quality_min": "candidate_quality_min",
            "active_quality_min": "active_quality_min",
            "oos_sharpe_min": "candidate_quality_min",
            "oos_quality_min": "candidate_quality_min",
            "probation_sharpe_min": "active_quality_min",
            "probation_quality_min": "active_quality_min",
            "dormant_sharpe_max": "active_quality_min",
            "dormant_quality_max": "active_quality_min",
            "dormant_revival_sharpe": "dormant_revival_quality",
        }
        for old, new in _LC_ALIASES.items():
            if old in lc_raw and new not in lc_raw:
                lc_raw[new] = lc_raw.pop(old)
        _f = cls._filter
        return cls(
            fitness_metric=raw.get("fitness_metric", "sharpe"),
            api=APIConfig(**_f(APIConfig, raw.get("api", {}))),
            generation=GenerationConfig(**_f(GenerationConfig, raw.get("generation", {}))),
            backtest=BacktestConfig(**_f(BacktestConfig, raw.get("backtest", {}))),
            validation=ValidationConfig(**_f(ValidationConfig, raw.get("validation", {}))),
            risk=RiskConfig(**_f(RiskConfig, raw.get("risk", {}))),
            trading=TradingConfig(**_f(TradingConfig, raw.get("trading", {}))),
            paper=PaperTradingConfig(**_f(PaperTradingConfig, raw.get("paper", {}))),
            forward=ForwardTestConfig(**_f(ForwardTestConfig, raw.get("forward", {}))),
            testnet=TestnetConfig(**_f(TestnetConfig, raw.get("testnet", {}))),
            event_driven=EventDrivenConfig(**_f(EventDrivenConfig, raw.get("event_driven", {}))),
            live_quality=LiveQualityConfig(**_f(LiveQualityConfig, raw.get("live_quality", {}))),
            deployment=DeploymentConfig(
                **_f(
                    DeploymentConfig,
                    raw.get("deployment", raw.get("universe", {})),
                )
            ),
            lifecycle=LifecycleTomlConfig(**_f(LifecycleTomlConfig, lc_raw)),
            execution=ExecutionTomlConfig(**_f(ExecutionTomlConfig, raw.get("execution", {}))),
            gate=GateTomlConfig(**_f(GateTomlConfig, raw.get("gate", {}))),
            regime=RegimeConfig(**_f(RegimeConfig, raw.get("regime", {}))),
            evo_daemon=EvoDaemonConfig(**_f(EvoDaemonConfig, raw.get("evo_daemon", {}))),
            admission=AdmissionConfig(**_f(AdmissionConfig, raw.get("admission", {}))),
            lifecycle_daemon=LifecycleDaemonConfig(**_f(LifecycleDaemonConfig, raw.get("lifecycle_daemon", {}))),
            stability=StabilityConfig(**_f(StabilityConfig, raw.get("stability", {}))),
        )
