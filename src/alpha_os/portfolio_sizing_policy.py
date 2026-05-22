from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, TypeAlias

import cvxpy as cp
import numpy as np
import pandas as pd

from .portfolio_decision import (
    CostInput,
    DependenceInput,
    ModelUncertaintyInput,
    PortfolioDecisionInput,
    PortfolioDecisionOutput,
    PortfolioTarget,
    PredictiveSignalInput,
    RiskInput,
    SizingDiagnostics,
    SizingRequest,
    SizingSolution,
    StructuralUncertaintyInput,
    UncertaintyInput,
)
from .portfolio_concentration import portfolio_effective_n, top_n_gross_share
from .portfolio_rebalance_friction import (
    PortfolioRebalanceFrictionPolicy,
    apply_portfolio_rebalance_friction,
)


@dataclass(frozen=True)
class SignalWeightedSizingPolicy:
    signal_scale: float = 1.0
    max_abs_weight: float = 1.0
    risk_aversion: float = 1.0
    model_uncertainty_aversion: float = 1.0
    structural_uncertainty_aversion: float = 1.0
    drawdown_aversion: float = 1.0
    uncertainty_aversion: float = 1.0
    dependence_aversion: float = 1.0


@dataclass(frozen=True)
class ConstrainedOptimizerSizingPolicy:
    signal_scale: float = 1.0
    max_abs_weight: float = 1.0
    risk_aversion: float = 1.0
    model_uncertainty_aversion: float = 1.0
    structural_uncertainty_aversion: float = 1.0
    drawdown_aversion: float = 1.0
    uncertainty_aversion: float = 1.0
    dependence_aversion: float = 1.0
    solver: str = "OSQP"


@dataclass(frozen=True)
class SignedMeanVarianceSizingPolicy:
    forecast_scale: float = 1.0
    max_abs_weight: float = 1.0
    risk_aversion: float = 1.0
    turnover_aversion: float = 1.0
    cost_aversion: float = 1.0
    short_cost_aversion: float = 1.0
    uncertainty_aversion: float = 1.0
    min_history_steps: int = 20
    covariance_shrinkage: float = 0.10
    solver: str = "OSQP"
    max_iterations: int = 200
    tolerance: float = 1e-6


@dataclass(frozen=True)
class HistoricalModelSizingPolicy:
    model_type: str = "hierarchical_risk_parity"
    min_history_steps: int = 20
    effective_n_floor: float | None = None
    top_gross_share_cap_n: int | None = None
    top_gross_share_cap: float | None = None


PortfolioSizingPolicy: TypeAlias = (
    SignalWeightedSizingPolicy
    | ConstrainedOptimizerSizingPolicy
    | SignedMeanVarianceSizingPolicy
    | HistoricalModelSizingPolicy
)


class PortfolioAllocator(Protocol):
    def allocate(self, request: SizingRequest) -> SizingSolution:
        ...


@dataclass(frozen=True)
class SignalWeightedAllocator:
    sizing_policy: SignalWeightedSizingPolicy

    def allocate(self, request: SizingRequest) -> SizingSolution:
        return apply_signal_weighted_allocation(
            request,
            sizing_policy=self.sizing_policy,
        )


@dataclass(frozen=True)
class ConstrainedOptimizerAllocator:
    sizing_policy: ConstrainedOptimizerSizingPolicy

    def allocate(self, request: SizingRequest) -> SizingSolution:
        return apply_constrained_optimizer_allocation(
            request,
            sizing_policy=self.sizing_policy,
        )


@dataclass(frozen=True)
class SignedMeanVarianceAllocator:
    sizing_policy: SignedMeanVarianceSizingPolicy

    def allocate(self, request: SizingRequest) -> SizingSolution:
        return apply_signed_mean_variance_allocation(
            request,
            sizing_policy=self.sizing_policy,
        )


@dataclass(frozen=True)
class HistoricalModelAllocator:
    sizing_policy: HistoricalModelSizingPolicy

    def allocate(self, request: SizingRequest) -> SizingSolution:
        return apply_historical_model_allocation(
            request,
            sizing_policy=self.sizing_policy,
        )


def portfolio_allocator_for_policy(
    sizing_policy: PortfolioSizingPolicy | None,
) -> PortfolioAllocator:
    if sizing_policy is None:
        return SignalWeightedAllocator(SignalWeightedSizingPolicy())
    if isinstance(sizing_policy, SignalWeightedSizingPolicy):
        return SignalWeightedAllocator(sizing_policy)
    if isinstance(sizing_policy, HistoricalModelSizingPolicy):
        return HistoricalModelAllocator(sizing_policy)
    if isinstance(sizing_policy, SignedMeanVarianceSizingPolicy):
        return SignedMeanVarianceAllocator(sizing_policy)
    if isinstance(sizing_policy, ConstrainedOptimizerSizingPolicy):
        return ConstrainedOptimizerAllocator(sizing_policy)
    raise TypeError(f"unsupported portfolio sizing policy: {type(sizing_policy).__name__}")


def apply_portfolio_sizing_policy(
    decision_input: PortfolioDecisionInput,
    sizing_policy: PortfolioSizingPolicy | None = None,
    rebalance_friction_policy: PortfolioRebalanceFrictionPolicy | None = None,
    apply_rebalance_friction: bool = True,
) -> PortfolioDecisionOutput:
    return apply_portfolio_allocator(
        decision_input,
        sizing_policy=sizing_policy,
        rebalance_friction_policy=rebalance_friction_policy,
        apply_rebalance_friction=apply_rebalance_friction,
    )


def apply_portfolio_allocator(
    decision_input: PortfolioDecisionInput,
    *,
    sizing_policy: PortfolioSizingPolicy | None = None,
    rebalance_friction_policy: PortfolioRebalanceFrictionPolicy | None = None,
    apply_rebalance_friction: bool = True,
) -> PortfolioDecisionOutput:
    request = build_sizing_request(decision_input)
    allocator = portfolio_allocator_for_policy(sizing_policy)
    solution = allocator.allocate(request)
    solution = _finalize_sizing_solution(
        request,
        solution,
        rebalance_friction_policy=rebalance_friction_policy,
        apply_rebalance_friction=apply_rebalance_friction,
    )
    return _portfolio_decision_output_from_solution(
        decision_input=decision_input,
        solution=solution,
    )


def build_sizing_request(decision_input: PortfolioDecisionInput) -> SizingRequest:
    subject_signals = _aggregate_subject_signals(decision_input.predictive_signals)
    current_weights = decision_input.portfolio_state.weights_by_subject
    subject_ids = tuple(sorted(set(current_weights) | set(subject_signals)))
    signal_horizons = _signal_horizons(decision_input.predictive_signals)
    dependence_matrix = _dependence_penalty_matrix(
        decision_input.dependence_inputs,
        subject_ids=subject_ids,
        dependence_aversion=1.0,
    )
    return SizingRequest(
        subject_ids=subject_ids,
        signal_values=tuple(subject_signals.get(subject_id, 0.0) for subject_id in subject_ids),
        current_weights=tuple(
            current_weights.get(subject_id, 0.0) for subject_id in subject_ids
        ),
        historical_return_matrix=_historical_return_matrix(
            decision_input.observed_inputs.historical_return_inputs,
            subject_ids=subject_ids,
        ),
        asset_classes=tuple(
            _subject_metadata_value(
                decision_input.subject_metadata_by_subject,
                subject_id,
                "asset_class",
            )
            for subject_id in subject_ids
        ),
        clusters=tuple(
            _subject_metadata_value(
                decision_input.subject_metadata_by_subject,
                subject_id,
                "cluster",
            )
            for subject_id in subject_ids
        ),
        uncertainty_std=tuple(
            _mean_uncertainty_std(decision_input.uncertainty_inputs, subject_id)
            for subject_id in subject_ids
        ),
        risk_values=tuple(
            _mean_risk_value(decision_input.risk_inputs, subject_id)
            for subject_id in subject_ids
        ),
        model_uncertainty_values=tuple(
            _mean_model_uncertainty_value(
                decision_input.model_uncertainty_inputs,
                subject_id,
            )
            for subject_id in subject_ids
        ),
        structural_uncertainty_values=tuple(
            _mean_structural_uncertainty_value(
                decision_input.structural_uncertainty_inputs,
                subject_id,
            )
            for subject_id in subject_ids
        ),
        dependence_values=tuple(
            _mean_dependence_value(decision_input.dependence_inputs, subject_id)
            for subject_id in subject_ids
        ),
        dependence_penalty_matrix=tuple(
            tuple(float(value) for value in row)
            for row in dependence_matrix
        ),
        no_trade_bands=tuple(
            _subject_cost_value(decision_input.cost_inputs, "no_trade_band", subject_id)
            for subject_id in subject_ids
        ),
        market_impact_levels=tuple(
            _cost_level(
                _subject_cost_value(
                    decision_input.cost_inputs,
                    "market_impact",
                    subject_id,
                )
            )
            for subject_id in subject_ids
        ),
        transaction_cost_levels=tuple(
            _transaction_cost_level(decision_input.cost_inputs, subject_id)
            for subject_id in subject_ids
        ),
        short_cost_levels=tuple(
            _short_cost_level(decision_input.cost_inputs, subject_id)
            for subject_id in subject_ids
        ),
        signal_horizons=tuple(
            signal_horizons.get(subject_id) for subject_id in subject_ids
        ),
        gross_exposure_cap=_resolved_gross_cap(decision_input),
        net_exposure_cap=_resolved_net_limit(decision_input),
        capital_base=max(decision_input.portfolio_state.capital_base, 0.0),
        holding_period_days=max(decision_input.portfolio_state.holding_period_days, 0),
        current_drawdown=max(decision_input.portfolio_state.current_drawdown, 0.0),
        recent_turnover=max(decision_input.portfolio_state.recent_turnover, 0.0),
        turnover_friction=_global_cost_value(
            decision_input.cost_inputs,
            "turnover_friction",
        ),
    )


def _subject_metadata_value(
    metadata_by_subject: dict[str, dict[str, str]],
    subject_id: str,
    key: str,
) -> str | None:
    value = metadata_by_subject.get(subject_id, {}).get(key)
    return value if isinstance(value, str) and value else None


def apply_signal_weighted_sizing(
    decision_input: PortfolioDecisionInput,
    sizing_policy: SignalWeightedSizingPolicy | None = None,
    rebalance_friction_policy: PortfolioRebalanceFrictionPolicy | None = None,
) -> PortfolioDecisionOutput:
    sizing_policy = sizing_policy or SignalWeightedSizingPolicy()
    return apply_portfolio_allocator(
        decision_input,
        sizing_policy=sizing_policy,
        rebalance_friction_policy=rebalance_friction_policy,
    )


def apply_signal_weighted_allocation(
    request: SizingRequest,
    sizing_policy: SignalWeightedSizingPolicy | None = None,
) -> SizingSolution:
    sizing_policy = sizing_policy or SignalWeightedSizingPolicy()
    subject_ids = request.subject_ids

    target_weights: list[float] = []
    risk_scales: list[float] = []
    for index, subject_id in enumerate(subject_ids):
        signal_value = request.signal_values[index]
        uncertainty_std = request.uncertainty_std[index]
        uncertainty_adjusted_signal = _uncertainty_adjusted_signal(
            signal_value,
            uncertainty_std=uncertainty_std,
            aversion=sizing_policy.uncertainty_aversion,
        )
        raw_weight = _clip(
            sizing_policy.signal_scale * uncertainty_adjusted_signal,
            -sizing_policy.max_abs_weight,
            sizing_policy.max_abs_weight,
        )

        risk_shrink = _shrink_from_level(
            request.risk_values[index],
            sizing_policy.risk_aversion,
        )
        model_uncertainty_shrink = _shrink_from_level(
            request.model_uncertainty_values[index],
            sizing_policy.model_uncertainty_aversion,
        )
        structural_uncertainty_shrink = _shrink_from_level(
            request.structural_uncertainty_values[index],
            sizing_policy.structural_uncertainty_aversion,
        )
        dependence_shrink = _shrink_from_level(
            request.dependence_values[index],
            sizing_policy.dependence_aversion,
        )
        drawdown_shrink = _shrink_from_level(
            request.current_drawdown,
            sizing_policy.drawdown_aversion,
        )
        risk_scale = (
            risk_shrink
            * model_uncertainty_shrink
            * structural_uncertainty_shrink
            * drawdown_shrink
            * dependence_shrink
        )
        target_weight = raw_weight * risk_scale

        target_weights.append(float(target_weight))
        risk_scales.append(float(risk_scale))
    return SizingSolution(
        subject_ids=subject_ids,
        target_weights=tuple(target_weights),
        risk_scales=tuple(risk_scales),
        diagnostics=SizingDiagnostics(
            backend_id="rule_based_signal_weighted",
            solver="-",
            status="ok",
        ),
    )


def apply_constrained_optimizer_sizing(
    decision_input: PortfolioDecisionInput,
    sizing_policy: ConstrainedOptimizerSizingPolicy | None = None,
    rebalance_friction_policy: PortfolioRebalanceFrictionPolicy | None = None,
) -> PortfolioDecisionOutput:
    sizing_policy = sizing_policy or ConstrainedOptimizerSizingPolicy()
    return apply_portfolio_allocator(
        decision_input,
        sizing_policy=sizing_policy,
        rebalance_friction_policy=rebalance_friction_policy,
    )


def apply_signed_mean_variance_sizing(
    decision_input: PortfolioDecisionInput,
    sizing_policy: SignedMeanVarianceSizingPolicy | None = None,
    rebalance_friction_policy: PortfolioRebalanceFrictionPolicy | None = None,
) -> PortfolioDecisionOutput:
    sizing_policy = sizing_policy or SignedMeanVarianceSizingPolicy()
    return apply_portfolio_allocator(
        decision_input,
        sizing_policy=sizing_policy,
        rebalance_friction_policy=rebalance_friction_policy,
    )


def apply_historical_model_sizing(
    decision_input: PortfolioDecisionInput,
    sizing_policy: HistoricalModelSizingPolicy | None = None,
    rebalance_friction_policy: PortfolioRebalanceFrictionPolicy | None = None,
) -> PortfolioDecisionOutput:
    sizing_policy = sizing_policy or HistoricalModelSizingPolicy()
    return apply_portfolio_allocator(
        decision_input,
        sizing_policy=sizing_policy,
        rebalance_friction_policy=rebalance_friction_policy,
    )


@dataclass(frozen=True)
class OptimizerSolveResult:
    weights: np.ndarray | None
    diagnostics: SizingDiagnostics


class SignedMeanVarianceOptimizerBackend(Protocol):
    def solve(
        self,
        request: SizingRequest,
        *,
        sizing_policy: SignedMeanVarianceSizingPolicy,
        gross_cap: float,
        risk_matrix: np.ndarray,
        alpha_vec: np.ndarray,
        current_weight_vec: np.ndarray,
        transaction_cost_vec: np.ndarray,
        short_cost_vec: np.ndarray,
    ) -> OptimizerSolveResult:
        ...


@dataclass(frozen=True)
class CvxpySignedMeanVarianceBackend:
    def solve(
        self,
        request: SizingRequest,
        *,
        sizing_policy: SignedMeanVarianceSizingPolicy,
        gross_cap: float,
        risk_matrix: np.ndarray,
        alpha_vec: np.ndarray,
        current_weight_vec: np.ndarray,
        transaction_cost_vec: np.ndarray,
        short_cost_vec: np.ndarray,
    ) -> OptimizerSolveResult:
        return _solve_signed_mean_variance_cvxpy(
            request,
            sizing_policy=sizing_policy,
            gross_cap=gross_cap,
            risk_matrix=risk_matrix,
            alpha_vec=alpha_vec,
            current_weight_vec=current_weight_vec,
            transaction_cost_vec=transaction_cost_vec,
            short_cost_vec=short_cost_vec,
        )


@dataclass(frozen=True)
class ProjectedGradientSignedMeanVarianceBackend:
    def solve(
        self,
        request: SizingRequest,
        *,
        sizing_policy: SignedMeanVarianceSizingPolicy,
        gross_cap: float,
        risk_matrix: np.ndarray,
        alpha_vec: np.ndarray,
        current_weight_vec: np.ndarray,
        transaction_cost_vec: np.ndarray,
        short_cost_vec: np.ndarray,
    ) -> OptimizerSolveResult:
        weights = _solve_signed_mean_variance_projected_gradient(
            request,
            sizing_policy=sizing_policy,
            gross_cap=gross_cap,
            risk_matrix=risk_matrix,
            alpha_vec=alpha_vec,
            current_weight_vec=current_weight_vec,
            transaction_cost_vec=transaction_cost_vec,
            short_cost_vec=short_cost_vec,
        )
        return OptimizerSolveResult(
            weights=weights,
            diagnostics=SizingDiagnostics(
                backend_id="projected_gradient_signed_mean_variance",
                solver=sizing_policy.solver,
                status="ok",
                objective_value=_signed_mean_variance_objective_value(
                    weights,
                    sizing_policy=sizing_policy,
                    risk_matrix=risk_matrix,
                    alpha_vec=alpha_vec,
                    current_weight_vec=current_weight_vec,
                    transaction_cost_vec=transaction_cost_vec,
                    short_cost_vec=short_cost_vec,
                ),
            ),
        )


def apply_signed_mean_variance_allocation(
    request: SizingRequest,
    sizing_policy: SignedMeanVarianceSizingPolicy | None = None,
) -> SizingSolution:
    sizing_policy = sizing_policy or SignedMeanVarianceSizingPolicy()
    subject_ids = request.subject_ids
    uses_cvxpy_backend = sizing_policy.solver.upper() in {"OSQP", "CLARABEL", "SCS"}
    if not subject_ids:
        return SizingSolution(
            subject_ids=(),
            target_weights=(),
            risk_scales=(),
            diagnostics=SizingDiagnostics(
                backend_id=(
                    "cvxpy_signed_mean_variance"
                    if uses_cvxpy_backend
                    else "projected_gradient_signed_mean_variance"
                ),
                solver=sizing_policy.solver,
                status="empty",
            ),
        )

    gross_cap = request.gross_exposure_cap
    if gross_cap is None or gross_cap <= 0.0:
        gross_cap = float(len(subject_ids)) * sizing_policy.max_abs_weight

    risk_matrix = _signed_optimizer_risk_matrix(
        request,
        min_history_steps=sizing_policy.min_history_steps,
        covariance_shrinkage=sizing_policy.covariance_shrinkage,
    )
    alpha_vec = np.asarray(
        [
            sizing_policy.forecast_scale
            * _uncertainty_adjusted_signal(
                signal_value,
                uncertainty_std=request.uncertainty_std[index],
                aversion=sizing_policy.uncertainty_aversion,
            )
            for index, signal_value in enumerate(request.signal_values)
        ],
        dtype=float,
    )
    current_weight_vec = np.asarray(request.current_weights, dtype=float)
    transaction_cost_vec = np.asarray(request.transaction_cost_levels, dtype=float)
    short_cost_vec = np.asarray(request.short_cost_levels, dtype=float)

    backend: SignedMeanVarianceOptimizerBackend
    if uses_cvxpy_backend:
        backend = CvxpySignedMeanVarianceBackend()
    else:
        backend = ProjectedGradientSignedMeanVarianceBackend()
    solve_result = backend.solve(
        request,
        sizing_policy=sizing_policy,
        gross_cap=gross_cap,
        risk_matrix=risk_matrix,
        alpha_vec=alpha_vec,
        current_weight_vec=current_weight_vec,
        transaction_cost_vec=transaction_cost_vec,
        short_cost_vec=short_cost_vec,
    )
    solved_weights = solve_result.weights
    diagnostics = solve_result.diagnostics
    if solved_weights is None:
        fallback_policy = replace(sizing_policy, solver="PROJECTED_GRADIENT")
        fallback_result = ProjectedGradientSignedMeanVarianceBackend().solve(
            request,
            sizing_policy=fallback_policy,
            gross_cap=gross_cap,
            risk_matrix=risk_matrix,
            alpha_vec=alpha_vec,
            current_weight_vec=current_weight_vec,
            transaction_cost_vec=transaction_cost_vec,
            short_cost_vec=short_cost_vec,
        )
        fallback_weights = (
            np.zeros(len(subject_ids), dtype=float)
            if fallback_result.weights is None
            else fallback_result.weights
        )
        return SizingSolution(
            subject_ids=subject_ids,
            target_weights=tuple(float(weight) for weight in fallback_weights),
            risk_scales=tuple(1.0 for _ in subject_ids),
            diagnostics=SizingDiagnostics(
                backend_id=(
                    f"{diagnostics.backend_id}->{fallback_result.diagnostics.backend_id}"
                ),
                solver=f"{sizing_policy.solver}->{fallback_policy.solver}",
                status="fallback",
                objective_value=fallback_result.diagnostics.objective_value,
                fallback_reason=diagnostics.fallback_reason or diagnostics.status,
            ),
        )
    return SizingSolution(
        subject_ids=subject_ids,
        target_weights=tuple(float(weight) for weight in solved_weights),
        risk_scales=tuple(1.0 for _ in subject_ids),
        diagnostics=diagnostics,
    )


def apply_constrained_optimizer_allocation(
    request: SizingRequest,
    sizing_policy: ConstrainedOptimizerSizingPolicy | None = None,
) -> SizingSolution:
    sizing_policy = sizing_policy or ConstrainedOptimizerSizingPolicy()
    subject_ids = request.subject_ids
    if not subject_ids:
        return SizingSolution(
            subject_ids=(),
            target_weights=(),
            risk_scales=(),
            diagnostics=SizingDiagnostics(
                backend_id="cvxpy_constrained_optimizer",
                solver=sizing_policy.solver,
                status="empty",
            ),
        )
    gross_cap = request.gross_exposure_cap
    if gross_cap is None or gross_cap <= 0.0:
        gross_cap = float(len(subject_ids)) * sizing_policy.max_abs_weight

    alpha = []
    risk_scales = []
    diagonal_penalties = []
    for index, subject_id in enumerate(subject_ids):
        signal_value = request.signal_values[index]
        uncertainty_std = request.uncertainty_std[index]
        adjusted_signal = _uncertainty_adjusted_signal(
            signal_value,
            uncertainty_std=uncertainty_std,
            aversion=sizing_policy.uncertainty_aversion,
        )
        risk_value = request.risk_values[index]
        model_uncertainty_value = request.model_uncertainty_values[index]
        structural_uncertainty_value = request.structural_uncertainty_values[index]
        dependence_value = request.dependence_values[index]
        risk_shrink = _shrink_from_level(risk_value, sizing_policy.risk_aversion)
        model_uncertainty_shrink = _shrink_from_level(
            model_uncertainty_value,
            sizing_policy.model_uncertainty_aversion,
        )
        structural_uncertainty_shrink = _shrink_from_level(
            structural_uncertainty_value,
            sizing_policy.structural_uncertainty_aversion,
        )
        dependence_shrink = _shrink_from_level(
            dependence_value,
            sizing_policy.dependence_aversion,
        )
        drawdown_shrink = _shrink_from_level(
            request.current_drawdown,
            sizing_policy.drawdown_aversion,
        )
        risk_scale = (
            risk_shrink
            * model_uncertainty_shrink
            * structural_uncertainty_shrink
            * dependence_shrink
            * drawdown_shrink
        )
        alpha.append(sizing_policy.signal_scale * adjusted_signal * risk_scale)
        risk_scales.append(risk_scale)
        diagonal_penalties.append(
            max(risk_value, 0.0) * sizing_policy.risk_aversion
            + max(model_uncertainty_value, 0.0)
            * sizing_policy.model_uncertainty_aversion
            + max(structural_uncertainty_value, 0.0)
            * sizing_policy.structural_uncertainty_aversion
            + max(dependence_value, 0.0) * sizing_policy.dependence_aversion
            + max(request.current_drawdown, 0.0) * sizing_policy.drawdown_aversion
            + 1e-6
        )

    alpha_vec = np.asarray(alpha, dtype=float)
    diagonal_penalty_vec = np.asarray(diagonal_penalties, dtype=float)
    dependence_matrix = np.asarray(request.dependence_penalty_matrix, dtype=float)
    dependence_matrix *= sizing_policy.dependence_aversion
    quadratic_penalty = np.diag(diagonal_penalty_vec) + dependence_matrix

    weights_var = cp.Variable(len(subject_ids))
    objective = cp.Maximize(
        alpha_vec @ weights_var
        - 0.5 * cp.quad_form(weights_var, quadratic_penalty)
    )
    constraints = [
        weights_var <= sizing_policy.max_abs_weight,
        weights_var >= -sizing_policy.max_abs_weight,
        cp.norm1(weights_var) <= gross_cap,
    ]
    if request.net_exposure_cap is not None and request.net_exposure_cap >= 0.0:
        constraints.append(cp.abs(cp.sum(weights_var)) <= request.net_exposure_cap)
    problem = cp.Problem(objective, constraints)
    solved = False
    solver_status = "not_solved"
    objective_value: float | None = None
    fallback_reason: str | None = None
    try:
        problem.solve(solver=sizing_policy.solver, warm_start=True)
        solver_status = str(problem.status)
        solved = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        objective_value = None if problem.value is None else float(problem.value)
    except cp.error.SolverError as error:
        solved = False
        solver_status = "solver_error"
        fallback_reason = type(error).__name__
    if not solved or weights_var.value is None:
        fallback_solution = apply_signal_weighted_allocation(
            request,
            sizing_policy=SignalWeightedSizingPolicy(
                signal_scale=sizing_policy.signal_scale,
                max_abs_weight=sizing_policy.max_abs_weight,
                risk_aversion=sizing_policy.risk_aversion,
                model_uncertainty_aversion=sizing_policy.model_uncertainty_aversion,
                structural_uncertainty_aversion=sizing_policy.structural_uncertainty_aversion,
                drawdown_aversion=sizing_policy.drawdown_aversion,
                uncertainty_aversion=sizing_policy.uncertainty_aversion,
                dependence_aversion=sizing_policy.dependence_aversion,
            ),
        )
        return SizingSolution(
            subject_ids=fallback_solution.subject_ids,
            target_weights=fallback_solution.target_weights,
            risk_scales=fallback_solution.risk_scales,
            diagnostics=SizingDiagnostics(
                backend_id=(
                    "cvxpy_constrained_optimizer"
                    f"->{fallback_solution.diagnostics.backend_id}"
                ),
                solver=sizing_policy.solver,
                status="fallback",
                objective_value=objective_value,
                fallback_reason=fallback_reason or solver_status,
            ),
        )

    solved_weights = np.asarray(weights_var.value, dtype=float).reshape(-1)
    return SizingSolution(
        subject_ids=subject_ids,
        target_weights=tuple(float(weight) for weight in solved_weights),
        risk_scales=tuple(risk_scales),
        diagnostics=SizingDiagnostics(
            backend_id="cvxpy_constrained_optimizer",
            solver=sizing_policy.solver,
            status=solver_status,
            objective_value=objective_value,
        ),
    )


def _solve_signed_mean_variance_cvxpy(
    request: SizingRequest,
    *,
    sizing_policy: SignedMeanVarianceSizingPolicy,
    gross_cap: float,
    risk_matrix: np.ndarray,
    alpha_vec: np.ndarray,
    current_weight_vec: np.ndarray,
    transaction_cost_vec: np.ndarray,
    short_cost_vec: np.ndarray,
) -> OptimizerSolveResult:
    weights_var = cp.Variable(len(request.subject_ids))
    objective = cp.Maximize(
        alpha_vec @ weights_var
        - 0.5
        * max(float(sizing_policy.risk_aversion), 0.0)
        * cp.quad_form(weights_var, cp.psd_wrap(risk_matrix))
        - max(float(sizing_policy.turnover_aversion), 0.0)
        * cp.sum_squares(weights_var - current_weight_vec)
        - max(float(sizing_policy.cost_aversion), 0.0)
        * transaction_cost_vec @ cp.abs(weights_var - current_weight_vec)
        - max(float(sizing_policy.short_cost_aversion), 0.0)
        * short_cost_vec @ cp.pos(-weights_var)
    )
    constraints = [
        weights_var <= sizing_policy.max_abs_weight,
        weights_var >= -sizing_policy.max_abs_weight,
        cp.norm1(weights_var) <= gross_cap,
    ]
    if request.net_exposure_cap is not None and request.net_exposure_cap >= 0.0:
        constraints.append(cp.abs(cp.sum(weights_var)) <= request.net_exposure_cap)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(
            solver=sizing_policy.solver,
            warm_start=True,
            **_cvxpy_signed_mean_variance_solver_options(sizing_policy.solver),
        )
    except cp.error.SolverError as error:
        return OptimizerSolveResult(
            weights=None,
            diagnostics=SizingDiagnostics(
                backend_id="cvxpy_signed_mean_variance",
                solver=sizing_policy.solver,
                status="solver_error",
                fallback_reason=type(error).__name__,
            ),
        )
    objective_value = None if problem.value is None else float(problem.value)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return OptimizerSolveResult(
            weights=None,
            diagnostics=SizingDiagnostics(
                backend_id="cvxpy_signed_mean_variance",
                solver=sizing_policy.solver,
                status=str(problem.status),
                objective_value=objective_value,
                fallback_reason=str(problem.status),
            ),
        )
    if weights_var.value is None:
        return OptimizerSolveResult(
            weights=None,
            diagnostics=SizingDiagnostics(
                backend_id="cvxpy_signed_mean_variance",
                solver=sizing_policy.solver,
                status="missing_solution",
                objective_value=objective_value,
                fallback_reason="missing_solution",
            ),
        )
    solved_weights = np.asarray(weights_var.value, dtype=float).reshape(-1)
    return OptimizerSolveResult(
        weights=_project_signed_optimizer_weights(
            solved_weights,
            max_abs_weight=sizing_policy.max_abs_weight,
            gross_cap=gross_cap,
            net_cap=request.net_exposure_cap,
        ),
        diagnostics=SizingDiagnostics(
            backend_id="cvxpy_signed_mean_variance",
            solver=sizing_policy.solver,
            status=str(problem.status),
            objective_value=objective_value,
        ),
    )


def _cvxpy_signed_mean_variance_solver_options(solver: str) -> dict[str, float | bool | int]:
    if solver.upper() == "OSQP":
        return {
            "eps_abs": 1e-4,
            "eps_rel": 1e-4,
            "max_iter": 2000,
            "polishing": False,
            "time_limit": 0.2,
        }
    return {}


def _solve_signed_mean_variance_projected_gradient(
    request: SizingRequest,
    *,
    sizing_policy: SignedMeanVarianceSizingPolicy,
    gross_cap: float,
    risk_matrix: np.ndarray,
    alpha_vec: np.ndarray,
    current_weight_vec: np.ndarray,
    transaction_cost_vec: np.ndarray,
    short_cost_vec: np.ndarray,
) -> np.ndarray:
    risk_aversion = max(float(sizing_policy.risk_aversion), 0.0)
    turnover_aversion = max(float(sizing_policy.turnover_aversion), 0.0)
    cost_aversion = max(float(sizing_policy.cost_aversion), 0.0)
    short_cost_aversion = max(float(sizing_policy.short_cost_aversion), 0.0)
    spectral_scale = 0.0
    if risk_matrix.size > 0:
        spectral_scale = float(np.max(np.linalg.eigvalsh(risk_matrix)))
    lipschitz = max(risk_aversion * spectral_scale + 2.0 * turnover_aversion, 1e-6)
    step_size = 1.0 / (lipschitz + 1.0)
    weights = _project_signed_optimizer_weights(
        current_weight_vec,
        max_abs_weight=sizing_policy.max_abs_weight,
        gross_cap=gross_cap,
        net_cap=request.net_exposure_cap,
    )
    max_iterations = max(int(sizing_policy.max_iterations), 1)
    tolerance = max(float(sizing_policy.tolerance), 0.0)
    for _ in range(max_iterations):
        previous = weights
        gradient = (
            risk_aversion * risk_matrix @ weights
            + 2.0 * turnover_aversion * (weights - current_weight_vec)
            - alpha_vec
        )
        if cost_aversion > 0.0:
            gradient = gradient + cost_aversion * transaction_cost_vec * np.sign(
                weights - current_weight_vec
            )
        if short_cost_aversion > 0.0:
            gradient = gradient + np.where(
                weights < 0.0,
                -short_cost_aversion * short_cost_vec,
                0.0,
            )
        weights = _project_signed_optimizer_weights(
            weights - step_size * gradient,
            max_abs_weight=sizing_policy.max_abs_weight,
            gross_cap=gross_cap,
            net_cap=request.net_exposure_cap,
        )
        if np.linalg.norm(weights - previous, ord=2) <= tolerance:
            break
    return weights


def _signed_mean_variance_objective_value(
    weights: np.ndarray,
    *,
    sizing_policy: SignedMeanVarianceSizingPolicy,
    risk_matrix: np.ndarray,
    alpha_vec: np.ndarray,
    current_weight_vec: np.ndarray,
    transaction_cost_vec: np.ndarray,
    short_cost_vec: np.ndarray,
) -> float:
    risk_aversion = max(float(sizing_policy.risk_aversion), 0.0)
    turnover_aversion = max(float(sizing_policy.turnover_aversion), 0.0)
    cost_aversion = max(float(sizing_policy.cost_aversion), 0.0)
    short_cost_aversion = max(float(sizing_policy.short_cost_aversion), 0.0)
    weight_vec = np.asarray(weights, dtype=float).reshape(-1)
    return float(
        alpha_vec @ weight_vec
        - 0.5 * risk_aversion * weight_vec @ risk_matrix @ weight_vec
        - turnover_aversion * np.sum(np.square(weight_vec - current_weight_vec))
        - cost_aversion
        * transaction_cost_vec
        @ np.abs(weight_vec - current_weight_vec)
        - short_cost_aversion * short_cost_vec @ np.maximum(-weight_vec, 0.0)
    )


def _project_signed_optimizer_weights(
    weights: np.ndarray,
    *,
    max_abs_weight: float,
    gross_cap: float,
    net_cap: float | None,
) -> np.ndarray:
    projected = np.nan_to_num(
        np.asarray(weights, dtype=float).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    max_abs = max(float(max_abs_weight), 0.0)
    gross_limit = max(float(gross_cap), 0.0)
    net_limit = None if net_cap is None else max(float(net_cap), 0.0)
    for _ in range(8):
        projected = np.clip(projected, -max_abs, max_abs)
        gross = float(np.sum(np.abs(projected)))
        if gross_limit > 0.0 and gross > gross_limit:
            projected = projected * (gross_limit / gross)
        if net_limit is not None:
            net = float(np.sum(projected))
            if abs(net) > net_limit and len(projected) > 0:
                projected = projected - (net - np.sign(net) * net_limit) / float(
                    len(projected)
                )
    projected = np.clip(projected, -max_abs, max_abs)
    gross = float(np.sum(np.abs(projected)))
    if gross_limit > 0.0 and gross > gross_limit:
        projected = projected * (gross_limit / gross)
    return projected


def apply_historical_model_allocation(
    request: SizingRequest,
    sizing_policy: HistoricalModelSizingPolicy | None = None,
) -> SizingSolution:
    sizing_policy = sizing_policy or HistoricalModelSizingPolicy()
    subject_ids = request.subject_ids
    if not subject_ids:
        return SizingSolution(
            subject_ids=(),
            target_weights=(),
            risk_scales=(),
            diagnostics=SizingDiagnostics(
                backend_id=f"skfolio:{sizing_policy.model_type}",
                solver="-",
                status="empty",
            ),
        )
    eligible_indices = _eligible_signal_indices(request.signal_values)
    if not eligible_indices:
        return _fallback_history_based_allocation(
            request,
            eligible_indices,
            fallback_reason="no_eligible_signals",
        )
    if sizing_policy.model_type == "equal_weight":
        return _fallback_history_based_allocation(
            request,
            eligible_indices,
            backend_id="history_based_equal_weight",
        )
    history = pd.DataFrame(
        request.historical_return_matrix,
        columns=subject_ids,
        dtype=float,
    )
    eligible_subject_ids = tuple(subject_ids[index] for index in eligible_indices)
    history = history.loc[:, list(eligible_subject_ids)]
    history = history.dropna(axis=1, how="all")
    history = history.dropna(axis=0, how="any")
    if history.empty or len(history.index) < max(int(sizing_policy.min_history_steps), 2):
        return _fallback_history_based_allocation(
            request,
            eligible_indices,
            fallback_reason="insufficient_history",
        )
    history = history.loc[:, history.var(axis=0) > 1e-12]
    if history.empty or len(history.columns) == 0:
        return _fallback_history_based_allocation(
            request,
            eligible_indices,
            fallback_reason="zero_variance_history",
        )

    fit_model_type = (
        "hierarchical_risk_parity"
        if sizing_policy.model_type
        in {
            "conviction_adjusted_hierarchical_risk_parity",
            "diversified_risk_budget",
        }
        else sizing_policy.model_type
    )
    try:
        weights = _fit_skfolio_sizing_method(
            history,
            model_type=fit_model_type,
        )
    except Exception as error:
        return _fallback_history_based_allocation(
            request,
            eligible_indices,
            fallback_reason=f"skfolio_error:{type(error).__name__}",
        )

    gross_cap = request.gross_exposure_cap
    if gross_cap is None or gross_cap <= 0.0:
        gross_cap = 1.0
    absolute_weights = np.abs(np.asarray(weights, dtype=float).reshape(-1))
    total_absolute_weight = float(np.sum(absolute_weights))
    if total_absolute_weight <= 0.0:
        return _fallback_history_based_allocation(
            request,
            eligible_indices,
            fallback_reason="non_positive_skfolio_weights",
        )
    scaled_weights = absolute_weights / total_absolute_weight * float(gross_cap)
    signal_by_subject = dict(zip(subject_ids, request.signal_values, strict=True))
    weight_map = {
        subject_id: 0.0
        for subject_id in subject_ids
    }
    for subject_id, weight in zip(history.columns, scaled_weights, strict=True):
        subject_key = str(subject_id)
        weight_map[subject_key] = float(weight) * _signal_direction(
            signal_by_subject[subject_key]
        )
    if sizing_policy.model_type == "conviction_adjusted_hierarchical_risk_parity":
        weight_map = _apply_directional_conviction_to_history_weights(
            request,
            weight_map=weight_map,
            history=history,
            gross_cap=float(gross_cap),
        )
    if sizing_policy.model_type == "diversified_risk_budget":
        weight_map = _apply_diversified_risk_budget_intent(
            request,
            weight_map=weight_map,
            gross_cap=float(gross_cap),
            effective_n_floor=sizing_policy.effective_n_floor,
            top_gross_share_cap_n=sizing_policy.top_gross_share_cap_n,
            top_gross_share_cap=sizing_policy.top_gross_share_cap,
        )
    return SizingSolution(
        subject_ids=subject_ids,
        target_weights=tuple(weight_map[subject_id] for subject_id in subject_ids),
        risk_scales=tuple(1.0 for _ in subject_ids),
        diagnostics=SizingDiagnostics(
            backend_id=f"skfolio:{fit_model_type}",
            solver="-",
            status="ok",
        ),
    )


def _apply_diversified_risk_budget_intent(
    request: SizingRequest,
    *,
    weight_map: dict[str, float],
    gross_cap: float,
    effective_n_floor: float | None,
    top_gross_share_cap_n: int | None,
    top_gross_share_cap: float | None,
) -> dict[str, float]:
    subject_ids = tuple(request.subject_ids)
    signs = {
        subject_id: _signal_direction(signal_value)
        for subject_id, signal_value in zip(
            request.subject_ids,
            request.signal_values,
            strict=True,
        )
    }
    eligible_subject_ids = tuple(
        subject_id
        for subject_id in subject_ids
        if signs.get(subject_id, 0.0) != 0.0
    )
    if not eligible_subject_ids:
        return {subject_id: 0.0 for subject_id in subject_ids}

    gross = sum(abs(float(weight_map.get(subject_id, 0.0))) for subject_id in subject_ids)
    if gross <= 1e-12:
        gross = max(float(gross_cap), 0.0)
    if gross <= 1e-12:
        return {subject_id: 0.0 for subject_id in subject_ids}
    gross = min(gross, max(float(gross_cap), 0.0)) if gross_cap > 0.0 else gross

    base_abs = {
        subject_id: abs(float(weight_map.get(subject_id, 0.0)))
        for subject_id in subject_ids
    }
    base_total = sum(base_abs.values())
    if base_total <= 1e-12:
        base_abs = {
            subject_id: gross / float(len(eligible_subject_ids))
            if subject_id in eligible_subject_ids
            else 0.0
            for subject_id in subject_ids
        }
    else:
        base_abs = {
            subject_id: value / base_total * gross
            for subject_id, value in base_abs.items()
        }
    equal_abs = {
        subject_id: gross / float(len(eligible_subject_ids))
        if subject_id in eligible_subject_ids
        else 0.0
        for subject_id in subject_ids
    }
    if _concentration_constraints_satisfied(
        base_abs,
        effective_n_floor=effective_n_floor,
        top_gross_share_cap_n=top_gross_share_cap_n,
        top_gross_share_cap=top_gross_share_cap,
    ):
        diversified_abs = base_abs
    else:
        low = 0.0
        high = 1.0
        for _ in range(32):
            midpoint = (low + high) / 2.0
            candidate = _blend_abs_weights(base_abs, equal_abs, midpoint)
            if _concentration_constraints_satisfied(
                candidate,
                effective_n_floor=effective_n_floor,
                top_gross_share_cap_n=top_gross_share_cap_n,
                top_gross_share_cap=top_gross_share_cap,
            ):
                high = midpoint
            else:
                low = midpoint
        diversified_abs = _blend_abs_weights(base_abs, equal_abs, high)
    return {
        subject_id: diversified_abs.get(subject_id, 0.0) * signs.get(subject_id, 0.0)
        for subject_id in subject_ids
    }


def _blend_abs_weights(
    base_abs: dict[str, float],
    equal_abs: dict[str, float],
    blend: float,
) -> dict[str, float]:
    blend = min(max(float(blend), 0.0), 1.0)
    return {
        subject_id: (1.0 - blend) * base_abs.get(subject_id, 0.0)
        + blend * equal_abs.get(subject_id, 0.0)
        for subject_id in base_abs
    }


def _concentration_constraints_satisfied(
    abs_weights_by_subject: dict[str, float],
    *,
    effective_n_floor: float | None,
    top_gross_share_cap_n: int | None,
    top_gross_share_cap: float | None,
) -> bool:
    weights = tuple(abs_weights_by_subject.values())
    if (
        effective_n_floor is not None
        and portfolio_effective_n(weights) + 1e-9 < float(effective_n_floor)
    ):
        return False
    if top_gross_share_cap_n is not None and top_gross_share_cap is not None:
        if top_n_gross_share(weights, top_n=int(top_gross_share_cap_n)) > (
            float(top_gross_share_cap) + 1e-9
        ):
            return False
    return True


def _apply_directional_conviction_to_history_weights(
    request: SizingRequest,
    *,
    weight_map: dict[str, float],
    history: pd.DataFrame,
    gross_cap: float,
) -> dict[str, float]:
    signal_by_subject = dict(zip(request.subject_ids, request.signal_values, strict=True))
    asset_class_by_subject = dict(zip(request.subject_ids, request.asset_classes, strict=True))
    long_abs_weights: dict[str, float] = {}
    short_abs_weights: dict[str, float] = {}
    for subject_id, weight in weight_map.items():
        signal_direction = _signal_direction(signal_by_subject.get(subject_id, 0.0))
        if signal_direction == 0.0 or weight == 0.0:
            continue
        abs_weight = abs(float(weight))
        if signal_direction > 0.0:
            long_abs_weights[subject_id] = abs_weight
        else:
            conviction = _short_directional_conviction(
                history[str(subject_id)] if subject_id in history.columns else None,
                asset_class=asset_class_by_subject.get(subject_id),
            )
            short_abs_weights[subject_id] = abs_weight * conviction

    long_total = sum(long_abs_weights.values())
    short_total = sum(short_abs_weights.values())
    if long_total <= 0.0 or short_total <= 0.0:
        return {subject_id: 0.0 for subject_id in request.subject_ids}

    side_budget = min(long_total, short_total, max(float(gross_cap), 0.0) / 2.0)
    adjusted: dict[str, float] = {subject_id: 0.0 for subject_id in request.subject_ids}
    for subject_id, abs_weight in long_abs_weights.items():
        adjusted[subject_id] = abs_weight / long_total * side_budget
    for subject_id, abs_weight in short_abs_weights.items():
        adjusted[subject_id] = -abs_weight / short_total * side_budget
    return adjusted


def _short_directional_conviction(
    returns: pd.Series | None,
    *,
    asset_class: str | None,
) -> float:
    asset_class_prior = {
        "equity_index": 0.35,
        "crypto": 0.25,
        "commodity": 0.70,
        "fx": 0.90,
        "rates": 0.90,
    }.get(asset_class or "", 0.75)
    trend_alignment = _short_trend_alignment(returns)
    breakout_penalty = _upward_breakout_short_penalty(returns)
    return float(asset_class_prior * trend_alignment * breakout_penalty)


def _short_trend_alignment(returns: pd.Series | None) -> float:
    return_63 = _window_cumulative_return(returns, 63)
    return_252 = _window_cumulative_return(returns, 252)
    negative_count = sum(value < 0.0 for value in (return_63, return_252))
    if negative_count == 2:
        return 1.0
    if negative_count == 1:
        return 0.65
    return 0.25


def _upward_breakout_short_penalty(returns: pd.Series | None) -> float:
    if returns is None or returns.empty:
        return 1.0
    recent = returns.astype(float).dropna().tail(63)
    if recent.empty:
        return 1.0
    return_63 = _window_cumulative_return(recent, 63)
    daily_vol = float(recent.std(ddof=0))
    threshold = max(0.10, 1.5 * daily_vol * np.sqrt(float(len(recent))))
    return 0.25 if return_63 > threshold else 1.0


def _window_cumulative_return(returns: pd.Series | None, window: int) -> float:
    if returns is None or returns.empty:
        return 0.0
    recent = returns.astype(float).dropna().tail(max(int(window), 1))
    if recent.empty:
        return 0.0
    return float(np.prod(1.0 + recent.to_numpy(dtype=float)) - 1.0)


def _aggregate_subject_signals(
    predictive_signals: tuple[PredictiveSignalInput, ...],
) -> dict[str, float]:
    observed_subject_ids: set[str] = set()
    weighted_values: dict[str, float] = {}
    weights: dict[str, float] = {}
    for signal in predictive_signals:
        observed_subject_ids.add(signal.subject_id)
        confidence = signal.confidence if signal.confidence is not None else 1.0
        confidence = max(confidence, 0.0)
        weighted_values[signal.subject_id] = (
            weighted_values.get(signal.subject_id, 0.0)
            + signal.value * confidence
        )
        weights[signal.subject_id] = weights.get(signal.subject_id, 0.0) + confidence
    return {
        subject_id: (
            float(weighted_values[subject_id] / weights[subject_id])
            if weights.get(subject_id, 0.0) > 0.0
            else 0.0
        )
        for subject_id in observed_subject_ids
    }


def _signal_horizons(
    predictive_signals: tuple[PredictiveSignalInput, ...],
) -> dict[str, int]:
    horizons: dict[str, list[int]] = {}
    for signal in predictive_signals:
        horizon = _target_horizon_days(signal.target_id)
        if horizon is None:
            continue
        horizons.setdefault(signal.subject_id, []).append(horizon)
    return {
        subject_id: min(values)
        for subject_id, values in horizons.items()
        if values
    }


def _dependence_penalty_matrix(
    dependence_inputs: tuple[DependenceInput, ...],
    *,
    subject_ids: tuple[str, ...],
    dependence_aversion: float,
) -> np.ndarray:
    matrix = np.zeros((len(subject_ids), len(subject_ids)), dtype=float)
    if dependence_aversion <= 0.0:
        return matrix
    index_by_subject = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    for item in dependence_inputs:
        left_index = index_by_subject.get(item.left_subject_id)
        right_index = index_by_subject.get(item.right_subject_id)
        if left_index is None or right_index is None:
            continue
        level = max(item.value, 0.0) * dependence_aversion
        if level <= 0.0:
            continue
        vector = np.zeros(len(subject_ids), dtype=float)
        vector[left_index] = 1.0
        vector[right_index] = 1.0
        matrix += level * np.outer(vector, vector)
    return matrix


def _mean_risk_value(
    risk_inputs: tuple[RiskInput, ...],
    subject_id: str,
) -> float:
    values = [
        max(risk_input.value, 0.0)
        for risk_input in risk_inputs
        if risk_input.subject_id == subject_id
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _mean_uncertainty_std(
    uncertainty_inputs: tuple[UncertaintyInput, ...],
    subject_id: str,
) -> float:
    values = [
        max(uncertainty_input.estimate_std, 0.0)
        for uncertainty_input in uncertainty_inputs
        if uncertainty_input.subject_id == subject_id
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _mean_model_uncertainty_value(
    model_uncertainty_inputs: tuple[ModelUncertaintyInput, ...],
    subject_id: str,
) -> float:
    values = [
        max(model_uncertainty_input.model_error, 0.0)
        for model_uncertainty_input in model_uncertainty_inputs
        if model_uncertainty_input.subject_id == subject_id
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _mean_structural_uncertainty_value(
    structural_uncertainty_inputs: tuple[StructuralUncertaintyInput, ...],
    subject_id: str,
) -> float:
    values = [
        max(structural_uncertainty_input.structural_error, 0.0)
        for structural_uncertainty_input in structural_uncertainty_inputs
        if structural_uncertainty_input.subject_id == subject_id
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _mean_dependence_value(
    dependence_inputs: tuple[DependenceInput, ...],
    subject_id: str,
) -> float:
    values = [
        max(dependence_input.value, 0.0)
        for dependence_input in dependence_inputs
        if dependence_input.left_subject_id == subject_id
        or dependence_input.right_subject_id == subject_id
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _subject_cost_value(
    cost_inputs: tuple[CostInput, ...],
    name: str,
    subject_id: str,
) -> float:
    subject_specific = [
        max(cost_input.value, 0.0)
        for cost_input in cost_inputs
        if cost_input.name == name and cost_input.subject_id == subject_id
    ]
    if subject_specific:
        return float(sum(subject_specific) / len(subject_specific))
    global_values = [
        max(cost_input.value, 0.0)
        for cost_input in cost_inputs
        if cost_input.name == name and cost_input.subject_id is None
    ]
    return float(sum(global_values) / len(global_values)) if global_values else 0.0


def _global_cost_value(
    cost_inputs: tuple[CostInput, ...],
    name: str,
) -> float:
    values = [
        max(cost_input.value, 0.0)
        for cost_input in cost_inputs
        if cost_input.name == name and cost_input.subject_id is None
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _global_risk_limit(
    risk_inputs: tuple[RiskInput, ...],
    name: str,
) -> float | None:
    values = [
        risk_input.value
        for risk_input in risk_inputs
        if risk_input.name == name and risk_input.subject_id is None
    ]
    if not values:
        return None
    return float(min(values))


def _resolved_gross_cap(decision_input: PortfolioDecisionInput) -> float | None:
    explicit_limit = _global_risk_limit(decision_input.risk_inputs, "gross_exposure_cap")
    if explicit_limit is not None:
        return explicit_limit
    return decision_input.portfolio_state.gross_limit


def _resolved_net_limit(decision_input: PortfolioDecisionInput) -> float | None:
    explicit_limit = _global_risk_limit(decision_input.risk_inputs, "net_exposure_cap")
    if explicit_limit is not None:
        return explicit_limit
    return decision_input.portfolio_state.net_limit


def _cost_level(value: float) -> float:
    return max(value / 10000.0, 0.0)


def _transaction_cost_level(
    cost_inputs: tuple[CostInput, ...],
    subject_id: str,
) -> float:
    return float(
        _cost_level(_subject_cost_value(cost_inputs, "market_impact", subject_id))
        + _cost_level(_subject_cost_value(cost_inputs, "fee_bps", subject_id))
        + _cost_level(_subject_cost_value(cost_inputs, "bid_ask_spread_bps", subject_id))
    )


def _short_cost_level(
    cost_inputs: tuple[CostInput, ...],
    subject_id: str,
) -> float:
    return float(
        _cost_level(_subject_cost_value(cost_inputs, "borrow_fee_bps_per_step", subject_id))
        + _cost_level(_subject_cost_value(cost_inputs, "borrow_fee_bps", subject_id))
    )


def _uncertainty_adjusted_signal(
    signal_value: float,
    *,
    uncertainty_std: float,
    aversion: float,
) -> float:
    if aversion <= 0.0 or uncertainty_std <= 0.0 or signal_value == 0.0:
        return float(signal_value)
    sign = 1.0 if signal_value > 0.0 else -1.0
    adjusted_magnitude = max(abs(signal_value) - aversion * uncertainty_std, 0.0)
    return float(sign * adjusted_magnitude)


def _shrink_from_level(level: float, aversion: float) -> float:
    if level <= 0.0 or aversion <= 0.0:
        return 1.0
    return float(1.0 / (1.0 + aversion * level))


def _apply_net_cap(
    targets: list[PortfolioTarget],
    net_limit: float,
    current_weights: dict[str, float],
    *,
    capital_base: float,
) -> list[PortfolioTarget]:
    net_exposure = float(sum(target.target_weight for target in targets))
    if abs(net_exposure) <= net_limit or abs(net_exposure) == 0.0:
        return targets
    scale = net_limit / abs(net_exposure)
    return [
        PortfolioTarget(
            subject_id=target.subject_id,
            target_weight=float(target.target_weight * scale),
            position_delta=float(target.target_weight * scale - current_weights.get(target.subject_id, 0.0)),
            target_notional=float(target.target_weight * scale * capital_base),
            target_quantity=target.target_quantity,
            entry_allowed=target.entry_allowed,
            risk_scale=target.risk_scale,
        )
        for target in targets
    ]


def _target_horizon_days(target_id: str) -> int | None:
    suffix = target_id.rsplit("_", 1)[-1]
    if not suffix.endswith("d"):
        return None
    value = suffix[:-1]
    if not value.isdigit():
        return None
    return int(value)


def _portfolio_targets_from_weights(
    subject_ids: tuple[str, ...],
    *,
    target_weights: tuple[float, ...],
    current_weights: dict[str, float],
    capital_base: float,
    risk_scales: tuple[float, ...],
) -> list[PortfolioTarget]:
    targets: list[PortfolioTarget] = []
    for subject_id, target_weight, risk_scale in zip(
        subject_ids,
        target_weights,
        risk_scales,
        strict=True,
    ):
        current_weight = current_weights.get(subject_id, 0.0)
        entry_allowed = not (abs(current_weight) == 0.0 and abs(target_weight) == 0.0)
        targets.append(
            PortfolioTarget(
                subject_id=subject_id,
                target_weight=float(target_weight),
                position_delta=float(target_weight - current_weight),
                target_notional=float(target_weight * capital_base),
                entry_allowed=entry_allowed,
                risk_scale=float(risk_scale),
            )
        )
    return targets


def _portfolio_decision_output_from_solution(
    *,
    decision_input: PortfolioDecisionInput,
    solution: SizingSolution,
) -> PortfolioDecisionOutput:
    targets = _portfolio_targets_from_weights(
        solution.subject_ids,
        target_weights=solution.target_weights,
        current_weights=decision_input.portfolio_state.weights_by_subject,
        capital_base=max(decision_input.portfolio_state.capital_base, 0.0),
        risk_scales=solution.risk_scales,
    )
    return PortfolioDecisionOutput(
        portfolio_id=decision_input.portfolio_id,
        as_of=decision_input.as_of,
        targets=tuple(targets),
        sizing_diagnostics=solution.diagnostics,
    )


def _historical_return_matrix(
    historical_return_inputs,
    *,
    subject_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    by_subject = {
        item.subject_id: item.returns_by_date
        for item in historical_return_inputs
    }
    common_dates: set[str] | None = None
    for subject_id in subject_ids:
        series = by_subject.get(subject_id)
        if not series:
            continue
        dates = set(series)
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    if not common_dates:
        return ()
    ordered_dates = sorted(common_dates)
    return tuple(
        tuple(
            float(by_subject[subject_id][date])
            if subject_id in by_subject
            else float("nan")
            for subject_id in subject_ids
        )
        for date in ordered_dates
    )


def _eligible_signal_indices(signal_values: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(signal_values)
        if abs(float(value)) > 0.0
    )


def _signal_direction(value: float) -> float:
    value = float(value)
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _signed_optimizer_risk_matrix(
    request: SizingRequest,
    *,
    min_history_steps: int,
    covariance_shrinkage: float,
) -> np.ndarray:
    size = len(request.subject_ids)
    if size <= 0:
        return np.zeros((0, 0), dtype=float)
    history = np.asarray(request.historical_return_matrix, dtype=float)
    if history.ndim != 2 or history.shape[1] != size:
        history = np.empty((0, size), dtype=float)
    if history.size > 0:
        history = history[np.isfinite(history).all(axis=1)]
    if len(history) >= max(int(min_history_steps), 2):
        covariance = np.cov(history, rowvar=False)
        covariance = np.asarray(covariance, dtype=float).reshape(size, size)
    else:
        risk_values = np.asarray(request.risk_values, dtype=float)
        variances = np.where(risk_values > 0.0, risk_values**2, 1.0)
        covariance = np.diag(variances)
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = (covariance + covariance.T) / 2.0
    diagonal = np.diag(np.diag(covariance))
    shrinkage = _clip(float(covariance_shrinkage), 0.0, 1.0)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    covariance += np.eye(size, dtype=float) * 1e-8
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if min_eigenvalue < 0.0:
        covariance += np.eye(size, dtype=float) * abs(min_eigenvalue)
    return covariance


def _fallback_history_based_allocation(
    request: SizingRequest,
    eligible_indices: tuple[int, ...],
    *,
    backend_id: str = "history_based_signal_equal_weight",
    fallback_reason: str | None = None,
) -> SizingSolution:
    subject_ids = request.subject_ids
    if not subject_ids or not eligible_indices:
        return SizingSolution(
            subject_ids=subject_ids,
            target_weights=tuple(0.0 for _ in subject_ids),
            risk_scales=tuple(1.0 for _ in subject_ids),
            diagnostics=SizingDiagnostics(
                backend_id=backend_id,
                solver="-",
                status="fallback" if fallback_reason is not None else "ok",
                fallback_reason=fallback_reason,
            ),
        )
    gross_cap = request.gross_exposure_cap
    if gross_cap is None or gross_cap <= 0.0:
        gross_cap = 1.0
    weight_per_subject = float(gross_cap) / float(len(eligible_indices))
    weights = [0.0 for _ in subject_ids]
    for index in eligible_indices:
        weights[index] = weight_per_subject * _signal_direction(
            request.signal_values[index]
        )
    return SizingSolution(
        subject_ids=subject_ids,
        target_weights=tuple(weights),
        risk_scales=tuple(1.0 for _ in subject_ids),
        diagnostics=SizingDiagnostics(
            backend_id=backend_id,
            solver="-",
            status="fallback" if fallback_reason is not None else "ok",
            fallback_reason=fallback_reason,
        ),
    )


def _finalize_sizing_solution(
    request: SizingRequest,
    solution: SizingSolution,
    *,
    rebalance_friction_policy: PortfolioRebalanceFrictionPolicy | None = None,
    apply_rebalance_friction: bool = True,
) -> SizingSolution:
    current_weights_by_subject = dict(
        zip(solution.subject_ids, request.current_weights, strict=True)
    )
    targets = _portfolio_targets_from_weights(
        solution.subject_ids,
        target_weights=solution.target_weights,
        current_weights=current_weights_by_subject,
        capital_base=request.capital_base,
        risk_scales=solution.risk_scales,
    )
    if apply_rebalance_friction:
        targets = apply_portfolio_rebalance_friction(
            request,
            targets,
            rebalance_friction_policy=rebalance_friction_policy,
        )
    if request.gross_exposure_cap is not None and request.gross_exposure_cap > 0.0:
        targets = _apply_gross_cap(
            targets,
            request.gross_exposure_cap,
            current_weights_by_subject,
        )
    if request.net_exposure_cap is not None and request.net_exposure_cap >= 0.0:
        targets = _apply_net_cap(
            targets,
            request.net_exposure_cap,
            current_weights_by_subject,
            capital_base=request.capital_base,
        )
    return SizingSolution(
        subject_ids=solution.subject_ids,
        target_weights=tuple(item.target_weight for item in targets),
        risk_scales=tuple(item.risk_scale for item in targets),
        diagnostics=solution.diagnostics,
    )


def _fit_skfolio_sizing_method(
    history: pd.DataFrame,
    *,
    model_type: str,
) -> np.ndarray:
    from skfolio.optimization import HierarchicalRiskParity, MeanRisk, RiskBudgeting

    if model_type == "equal_weight":
        weights = np.repeat(1.0 / float(len(history.columns)), len(history.columns))
    elif model_type == "minimum_variance":
        estimator = MeanRisk()
        estimator.fit(history)
        weights = estimator.weights_
    elif model_type == "risk_budgeting":
        estimator = RiskBudgeting()
        estimator.fit(history)
        weights = estimator.weights_
    elif model_type == "hierarchical_risk_parity":
        estimator = HierarchicalRiskParity()
        estimator.fit(history)
        weights = estimator.weights_
    else:
        raise ValueError(f"unknown skfolio model_type: {model_type}")
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("skfolio returned non-positive total weight")
    return weights / total


def _apply_gross_cap(
    targets: list[PortfolioTarget],
    gross_cap: float,
    current_weights: dict[str, float],
) -> list[PortfolioTarget]:
    gross = float(sum(abs(target.target_weight) for target in targets))
    if gross <= gross_cap or gross == 0.0:
        return targets
    scale = gross_cap / gross
    capped_targets: list[PortfolioTarget] = []
    for target in targets:
        capped_targets.append(
            PortfolioTarget(
                subject_id=target.subject_id,
                target_weight=float(target.target_weight * scale),
                position_delta=float(
                    target.target_weight * scale
                    - current_weights.get(target.subject_id, 0.0)
                ),
                target_notional=(
                    None
                    if target.target_notional is None
                    else float(target.target_notional * scale)
                ),
                target_quantity=target.target_quantity,
                entry_allowed=target.entry_allowed,
                risk_scale=target.risk_scale,
            )
        )
    return capped_targets


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))
