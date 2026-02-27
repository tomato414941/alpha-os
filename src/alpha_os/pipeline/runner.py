"""Pipeline runner — generate → validate → combine → trade integrated loop."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from ..alpha.combiner import (
    WeightedCombinerConfig,
    compute_diversity_scores,
    compute_weights,
    weighted_combine,
)
from ..alpha.evaluator import evaluate_expression, normalize_signal
from ..alpha.registry import AlphaRecord, AlphaRegistry, AlphaState
from ..backtest.cost_model import CostModel
from ..backtest.engine import BacktestEngine
from ..dsl import to_string
from ..dsl.expr import Expr
from ..evolution.archive import AlphaArchive
from ..evolution.behavior import compute_behavior
from ..evolution.gp import GPConfig, GPEvolver
from ..governance.gates import GateConfig, adoption_gate
from ..validation.deflated_sharpe import deflated_sharpe_ratio
from ..validation.pbo import probability_of_backtest_overfitting
from ..validation.purged_cv import purged_walk_forward

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    gp: GPConfig | None = None
    combiner_weighted: WeightedCombinerConfig | None = None
    gate: GateConfig | None = None
    commission_pct: float = 0.10
    slippage_pct: float = 0.05
    n_cv_folds: int = 5
    embargo_days: int = 5
    eval_window_days: int = 0  # 0 = all data; >0 = trailing N days


@dataclass
class PipelineResult:
    n_generated: int
    n_validated: int
    n_adopted: int
    n_combined: int
    combined_signal: np.ndarray | None
    elapsed: float
    archive_coverage: float


class PipelineRunner:
    """Orchestrates the full alpha generation pipeline."""

    def __init__(
        self,
        features: list[str],
        data: dict[str, np.ndarray],
        prices: np.ndarray,
        config: PipelineConfig | None = None,
        registry: AlphaRegistry | None = None,
        seed: int = 42,
    ):
        self.features = features
        self.config = config or PipelineConfig()
        self.registry = registry
        self.seed = seed

        # Apply evaluation window
        window = self.config.eval_window_days
        if window > 0 and len(prices) > window:
            self.data = {k: v[-window:] for k, v in data.items()}
            self.prices = prices[-window:]
            logger.info("Eval window: using last %d of %d days", window, len(prices))
        else:
            self.data = data
            self.prices = prices

        gate_cfg = self.config.gate or GateConfig()
        if window > 0 and window < gate_cfg.min_n_days:
            logger.warning(
                "eval_window_days=%d < min_n_days=%d — all alphas will fail the n_days gate",
                window, gate_cfg.min_n_days,
            )

        self.engine = BacktestEngine(
            CostModel(self.config.commission_pct, self.config.slippage_pct)
        )
        self.archive = AlphaArchive()

    def run(self) -> PipelineResult:
        """Execute full pipeline: evolve → validate → adopt → combine."""
        t0 = time.perf_counter()
        cfg = self.config
        n_days = len(self.prices)

        # Phase 1: Evolve
        logger.info("Phase 1: Evolving alphas...")
        results = self._evolve()
        n_generated = len(results)
        logger.info(f"  Generated {n_generated} unique alphas")

        # Phase 2: Evaluate + validate
        logger.info("Phase 2: Validating alphas...")
        validated = self._validate(results)
        n_validated = len(validated)
        logger.info(f"  {n_validated} alphas passed validation")

        # Phase 3: Adopt via governance gates
        logger.info("Phase 3: Governance gates...")
        adopted = self._adopt(validated)
        n_adopted = len(adopted)
        logger.info(f"  {n_adopted} alphas adopted")

        # Phase 4: Combine
        logger.info("Phase 4: Combining alphas...")
        combined_signal = None
        if adopted:
            combined_signal = self._combine(adopted)
            logger.info(f"  Combined {len(adopted)} alphas into portfolio signal")

        elapsed = time.perf_counter() - t0
        return PipelineResult(
            n_generated=n_generated,
            n_validated=n_validated,
            n_adopted=n_adopted,
            n_combined=len(adopted),
            combined_signal=combined_signal,
            elapsed=elapsed,
            archive_coverage=self.archive.coverage,
        )

    def _evolve(self) -> list[tuple[Expr, float]]:
        data = self.data
        prices = self.prices
        n_days = len(prices)

        def evaluate_fn(expr):
            try:
                sig = evaluate_expression(expr, data, n_days)
                result = self.engine.run(sig, prices)
                return result.sharpe
            except Exception:
                return -999.0

        gp_cfg = self.config.gp or GPConfig()
        evolver = GPEvolver(self.features, evaluate_fn, config=gp_cfg, seed=self.seed)
        results = evolver.run()

        # Fill archive
        live_signals: list[np.ndarray] = []
        for expr, fitness in results:
            try:
                sig = evaluate_expression(expr, data, n_days)
                behavior = compute_behavior(sig, expr, live_signals)
                if self.archive.add(expr, fitness, behavior):
                    live_signals.append(sig)
            except Exception:
                continue

        return results

    def _validate(
        self, results: list[tuple[Expr, float]]
    ) -> list[tuple[Expr, float, float, float]]:
        """Returns (expr, fitness, oos_sharpe, dsr_pvalue) for passing alphas."""
        cfg = self.config
        n_days = len(self.prices)
        n_trials = len(results)
        validated = []

        for expr, fitness in results:
            if fitness <= 0:
                continue
            try:
                sig = evaluate_expression(expr, self.data, n_days)

                # Purged WF CV
                cv = purged_walk_forward(
                    sig, self.prices, self.engine,
                    n_folds=cfg.n_cv_folds, embargo=cfg.embargo_days,
                )
                if cv.oos_sharpe <= 0:
                    continue

                # DSR
                bt = self.engine.run(sig, self.prices)
                pos = normalize_signal(sig)
                rets = np.diff(self.prices) / self.prices[:-1]
                n = min(len(pos) - 1, len(rets))
                strat_rets = pos[:n] * rets[:n]
                dsr = deflated_sharpe_ratio(strat_rets, n_trials=n_trials)

                validated.append((expr, fitness, cv.oos_sharpe, dsr.p_value))
            except Exception:
                continue

        return validated

    def _compute_batch_pbo(
        self, validated: list[tuple[Expr, float, float, float]],
        max_pbo_signals: int = 200,
    ) -> float:
        """Compute batch PBO for validated alphas. Returns PBO in [0, 1]."""
        n_days = len(self.prices)
        signals = []
        for expr, fitness, oos_sharpe, dsr_pvalue in validated:
            try:
                sig = evaluate_expression(expr, self.data, n_days)
                signals.append(sig)
            except Exception:
                continue

        if len(signals) < 2:
            return 1.0

        # Sample if too many signals (PBO cost scales with n_strategies)
        if len(signals) > max_pbo_signals:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(signals), max_pbo_signals, replace=False)
            signals = [signals[i] for i in indices]

        sig_matrix = np.array(signals)
        pbo_result = probability_of_backtest_overfitting(
            sig_matrix, self.prices, self.engine,
            n_blocks=10, max_combinations=50,
        )
        logger.info(
            "  Batch PBO: %.3f (%d strategies, %d combinations)",
            pbo_result.pbo, len(signals), pbo_result.n_combinations,
        )
        return pbo_result.pbo

    def _adopt(
        self, validated: list[tuple[Expr, float, float, float]]
    ) -> list[tuple[Expr, float]]:
        """Apply governance gates and register adopted alphas."""
        gate_cfg = self.config.gate or GateConfig()

        # Compute batch PBO once for all validated alphas
        batch_pbo = self._compute_batch_pbo(validated)

        adopted = []
        for expr, fitness, oos_sharpe, dsr_pvalue in validated:
            result = adoption_gate(
                oos_sharpe=oos_sharpe,
                pbo=batch_pbo,
                dsr_pvalue=dsr_pvalue,
                fdr_passed=True,
                avg_correlation=0.0,
                n_days=len(self.prices),
                config=gate_cfg,
            )
            if result.passed:
                adopted.append((expr, fitness))
                if self.registry:
                    record = AlphaRecord(
                        alpha_id=f"alpha_{hash(repr(expr)) % 10**8:08d}",
                        expression=to_string(expr),
                        state=AlphaState.ACTIVE,
                        fitness=fitness,
                        oos_sharpe=oos_sharpe,
                        pbo=batch_pbo,
                        dsr_pvalue=dsr_pvalue,
                    )
                    self.registry.register(record)

        return adopted

    def _combine(self, adopted: list[tuple[Expr, float]]) -> np.ndarray:
        """Build combined signal using quality × diversity weighting."""
        n_days = len(self.prices)
        signals = []
        sharpes = []

        for expr, fitness in adopted:
            try:
                sig = evaluate_expression(expr, self.data, n_days)
                signals.append(sig)
                sharpes.append(fitness)
            except Exception:
                continue

        if not signals:
            return np.zeros(n_days)

        sig_matrix = np.array(signals)
        sharpe_arr = np.array(sharpes)

        wcfg = self.config.combiner_weighted or WeightedCombinerConfig()
        diversity = compute_diversity_scores(sig_matrix, chunk_size=wcfg.chunk_size)
        weights = compute_weights(sharpe_arr, diversity, min_weight=wcfg.min_weight)
        return weighted_combine(sig_matrix, weights)
