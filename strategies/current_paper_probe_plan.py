from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PaperProbePlanRow:
    rank: int
    opportunity: str
    probe_type: str
    status: str
    side: str
    priority_score: float
    asset: str
    venue: str
    candidate_size_usd: str
    observation_horizon: str
    evidence: str
    missing_evidence: str
    next_step: str


def build_paper_probe_plan(
    *,
    stack_path: Path = ROOT / "current_alpha_stack.csv",
    top: int = 30,
) -> tuple[PaperProbePlanRow, ...]:
    candidates = tuple(
        row
        for row in _read_rows(stack_path)
        if _probe_type(row) != ""
    )
    sorted_candidates = sorted(candidates, key=lambda row: _float(row.get("priority_score")), reverse=True)
    return tuple(
        _build_plan_row(rank=index + 1, row=row)
        for index, row in enumerate(sorted_candidates[:top])
    )


def write_paper_probe_plan_csv(
    rows: tuple[PaperProbePlanRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "rank",
                "opportunity",
                "probe_type",
                "status",
                "side",
                "priority_score",
                "asset",
                "venue",
                "candidate_size_usd",
                "observation_horizon",
                "evidence",
                "missing_evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.rank,
                    row.opportunity,
                    row.probe_type,
                    row.status,
                    row.side,
                    f"{row.priority_score:.8f}",
                    row.asset,
                    row.venue,
                    row.candidate_size_usd,
                    row.observation_horizon,
                    row.evidence,
                    row.missing_evidence,
                    row.next_step,
                )
            )
    return output_path


def write_paper_probe_plan_md(
    rows: tuple[PaperProbePlanRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Paper Probe Plan\n\n")
        handle.write(
            "This is the current cross-lane queue for small paper observations. "
            "It is not a trade instruction, not a live execution system, and not "
            "a deployable strategy list.\n\n"
        )
        handle.write(
            "| rank | opportunity | probe type | side | priority | asset | venue | size USD | horizon | missing evidence | next step |\n"
        )
        handle.write("| ---: | --- | --- | --- | ---: | --- | --- | ---: | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.rank} | "
                f"{row.opportunity} | "
                f"{row.probe_type} | "
                f"{row.side} | "
                f"{row.priority_score:.4f} | "
                f"{row.asset} | "
                f"{row.venue} | "
                f"{row.candidate_size_usd} | "
                f"{row.observation_horizon} | "
                f"{_escape(row.missing_evidence)} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The queue promotes candidates only when another screen has already "
            "found a label or rough execution gate. The remaining work is to record "
            "fresh paper observations with fill, funding, stop, and adverse-selection "
            "evidence where the venue supports it.\n"
        )
    return output_path


def _build_plan_row(*, rank: int, row: dict[str, str]) -> PaperProbePlanRow:
    evidence = row.get("evidence", "")
    return PaperProbePlanRow(
        rank=rank,
        opportunity=row.get("opportunity", ""),
        probe_type=_probe_type(row),
        status=row.get("status", ""),
        side=row.get("side", ""),
        priority_score=_float(row.get("priority_score")),
        asset=_asset(evidence=evidence, opportunity=row.get("opportunity", "")),
        venue=_venue(evidence=evidence, sources=row.get("sources", "")),
        candidate_size_usd=_candidate_size(row),
        observation_horizon=_observation_horizon(row),
        evidence=evidence,
        missing_evidence=_missing_evidence(row.get("conflict", "")),
        next_step=row.get("next_step", ""),
    )


def _probe_type(row: dict[str, str]) -> str:
    status = row.get("status", "")
    text = " ".join((status, row.get("opportunity", ""), row.get("side", ""), row.get("next_step", ""))).lower()
    if status == "small_repeat_paper_check":
        return "repeat_execution_probe"
    if status == "microstructure_small_paper_probe":
        return "microstructure_flow_probe"
    if status == "volume_dislocation_execution_probe":
        return "volume_dislocation_probe"
    if status in {
        "paper_outcome_supported_carry_reversion_probe",
        "paper_short_horizon_supported_carry_reversion_probe",
        "paper_executable_carry_reversion_probe",
        "paper_delayed_carry_reversion_probe",
    }:
        return "crowding_reversion_probe"
    if status == "small_paper_probe":
        return "liquidation_intensity_probe"
    if status == "low_cost_intraday_paper_supported":
        return "intraday_derivatives_probe"
    if status == "dislocation_repeat_execution_candidate":
        return "dislocation_repeat_probe"
    if "paper-check" in text and "candidate_after_refresh_check" in status:
        return "event_probability_probe"
    return ""


def _asset(*, evidence: str, opportunity: str) -> str:
    if ":" in evidence:
        return evidence.split(":", 1)[0].strip()
    return opportunity.split("_", 1)[0].upper()


def _venue(*, evidence: str, sources: str) -> str:
    match = re.search(r"\bvenue=([^,\s]+)", evidence)
    if match:
        return match.group(1)
    if "source=hyperliquid" in evidence.lower() or "hyperliquid" in sources.lower():
        return "HL"
    if "OKX" in evidence or "okx" in sources.lower():
        return "OKX"
    if "prediction_markets" in sources:
        return "prediction_market"
    return ""


def _candidate_size(row: dict[str, str]) -> str:
    evidence = row.get("evidence", "")
    match = re.search(r"\bsize=([0-9]+(?:\.[0-9]+)?)", evidence)
    if match:
        return match.group(1)
    match = re.search(r"\bdepth_usage_([0-9]+)=", evidence)
    if match:
        return match.group(1)
    if row.get("status") == "small_repeat_paper_check":
        return "1000"
    return ""


def _observation_horizon(row: dict[str, str]) -> str:
    text = " ".join((row.get("evidence", ""), row.get("next_step", ""))).lower()
    horizons = tuple(horizon for horizon in ("15m", "1h", "4h", "12h", "24h") if horizon in text)
    return "/".join(horizons) if horizons else "fresh"


def _missing_evidence(conflict: str) -> str:
    parts = [part.strip() for part in re.split(r";|,", conflict) if part.strip()]
    return "; ".join(parts[:4]) if parts else conflict


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-path", type=Path, default=ROOT / "current_alpha_stack.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_paper_probe_plan.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_paper_probe_plan.md")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_paper_probe_plan(stack_path=args.stack_path, top=args.top)
    write_paper_probe_plan_csv(rows, output_path=args.output_path)
    write_paper_probe_plan_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.rank, row.opportunity, row.probe_type, row.side, f"priority={row.priority_score:.4f}")


if __name__ == "__main__":
    main()
