from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExecutionScenario:
    name: str
    spot_leg_cost_bps: float
    perp_leg_cost_bps: float


DEFAULT_SCENARIOS = (
    ExecutionScenario(
        name="low_slippage_maker_like",
        spot_leg_cost_bps=10.0,
        perp_leg_cost_bps=2.0,
    ),
    ExecutionScenario(
        name="low_slippage_taker_like",
        spot_leg_cost_bps=10.0,
        perp_leg_cost_bps=5.0,
    ),
    ExecutionScenario(
        name="retail_taker_with_slippage",
        spot_leg_cost_bps=15.0,
        perp_leg_cost_bps=7.5,
    ),
    ExecutionScenario(
        name="expensive_or_thin_execution",
        spot_leg_cost_bps=20.0,
        perp_leg_cost_bps=10.0,
    ),
)


@dataclass(frozen=True)
class ExecutionGate:
    candidate: str
    scenario: str
    max_paired_leg_cost_bps: float
    scenario_paired_leg_cost_bps: float
    headroom_bps: float
    passes: bool
    default_cost_total_return: float
    default_cost_sharpe: float
    mean_daily_turnover: float


def build_execution_gates(
    *,
    fee_ceiling_path: Path,
    scenarios: tuple[ExecutionScenario, ...] = DEFAULT_SCENARIOS,
) -> tuple[ExecutionGate, ...]:
    rows = _read_fee_ceiling_rows(fee_ceiling_path)
    gates = [
        _build_gate(row=row, scenario=scenario)
        for row in rows
        for scenario in scenarios
    ]
    return tuple(
        sorted(
            gates,
            key=lambda gate: (
                gate.passes,
                gate.headroom_bps,
                gate.default_cost_sharpe,
                gate.default_cost_total_return,
            ),
            reverse=True,
        )
    )


def write_execution_gates_csv(
    gates: tuple[ExecutionGate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "candidate",
                "scenario",
                "max_paired_leg_cost_bps",
                "scenario_paired_leg_cost_bps",
                "headroom_bps",
                "passes",
                "default_cost_total_return",
                "default_cost_sharpe",
                "mean_daily_turnover",
            )
        )
        for gate in gates:
            writer.writerow(
                (
                    gate.candidate,
                    gate.scenario,
                    f"{gate.max_paired_leg_cost_bps:.6f}",
                    f"{gate.scenario_paired_leg_cost_bps:.6f}",
                    f"{gate.headroom_bps:.6f}",
                    gate.passes,
                    f"{gate.default_cost_total_return:.10f}",
                    f"{gate.default_cost_sharpe:.10f}",
                    f"{gate.mean_daily_turnover:.10f}",
                )
            )
    return output_path


def write_execution_gates_md(
    gates: tuple[ExecutionGate, ...],
    *,
    output_path: Path,
    top: int = 16,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Spot/Perp Carry Execution Gate\n\n")
        handle.write(
            "This compares each carry candidate's fee ceiling with simple execution "
            "scenarios. The scenarios are assumptions, not exchange fee schedules.\n\n"
        )
        handle.write(
            "| candidate | scenario | ceiling bps | scenario bps | headroom bps | pass | default sharpe | turnover |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |\n")
        for gate in gates[:top]:
            handle.write(
                "| "
                f"{gate.candidate} | "
                f"{gate.scenario} | "
                f"{gate.max_paired_leg_cost_bps:.6f} | "
                f"{gate.scenario_paired_leg_cost_bps:.6f} | "
                f"{gate.headroom_bps:.6f} | "
                f"{gate.passes} | "
                f"{gate.default_cost_sharpe:.6f} | "
                f"{gate.mean_daily_turnover:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A candidate only graduates from historical carry screen to execution watch "
            "when it has positive headroom under at least one realistic fee/slippage "
            "scenario. The low-turnover 14-day cluster is the only current cluster with "
            "meaningful room.\n"
        )
    return output_path


def _build_gate(
    *,
    row: dict[str, str],
    scenario: ExecutionScenario,
) -> ExecutionGate:
    ceiling = float(row["max_paired_leg_cost_bps"])
    scenario_cost = (scenario.spot_leg_cost_bps + scenario.perp_leg_cost_bps) / 2.0
    headroom = ceiling - scenario_cost
    return ExecutionGate(
        candidate=row["candidate"],
        scenario=scenario.name,
        max_paired_leg_cost_bps=ceiling,
        scenario_paired_leg_cost_bps=scenario_cost,
        headroom_bps=headroom,
        passes=headroom > 0.0,
        default_cost_total_return=float(row["default_cost_total_return"]),
        default_cost_sharpe=float(row["default_cost_sharpe"]),
        mean_daily_turnover=float(row["mean_daily_turnover"]),
    )


def _read_fee_ceiling_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fee-ceiling-path",
        type=Path,
        default=ROOT / "spot_perp_carry_fee_ceiling.csv",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "spot_perp_carry_execution_gate.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "spot_perp_carry_execution_gate.md",
    )
    parser.add_argument("--top", type=int, default=16)
    args = parser.parse_args()

    gates = build_execution_gates(fee_ceiling_path=args.fee_ceiling_path)
    write_execution_gates_csv(gates, output_path=args.csv_output_path)
    write_execution_gates_md(gates, output_path=args.md_output_path, top=args.top)
    for gate in gates[: args.top]:
        print(
            gate.candidate,
            gate.scenario,
            f"headroom_bps={gate.headroom_bps:.6f}",
            f"pass={gate.passes}",
        )


if __name__ == "__main__":
    main()
