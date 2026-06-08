from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parents[0]


@dataclass(frozen=True)
class OfiCooldownStateGate:
    asset: str
    decision: str
    lifecycle_status: str
    second_repeat_5m_bps: str
    current_mark: str
    spread_bps: str
    near_depth_10bps_notional: str
    annualized_funding: str
    gate_action: str
    reason: str
    next_step: str


def build_ofi_cooldown_state_gate(
    *,
    timing_decay_path: Path = ROOT / "current_ofi_timing_decay_review.csv",
    hl_context_path: Path = STRATEGIES_ROOT / "candidate_validation" / "current_followup_execution_context.csv",
) -> tuple[OfiCooldownStateGate, ...]:
    context = {row.get("asset", ""): row for row in _read_rows(hl_context_path)}
    rows = tuple(_gate_for_row(row=row, context=context.get(row.get("asset", ""), {})) for row in _read_rows(timing_decay_path))
    return tuple(sorted(rows, key=lambda row: row.asset))


def write_ofi_cooldown_state_gate_csv(rows: tuple[OfiCooldownStateGate, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(OfiCooldownStateGate.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_ofi_cooldown_state_gate_md(rows: tuple[OfiCooldownStateGate, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OFI Cooldown State Gate\n\n")
        handle.write(
            "This blocks immediate OFI repeats after timing decay and records what fresh state is needed. "
            "It is a paper-observation gate, not a live trading rule.\n\n"
        )
        handle.write(f"Generated at: {generated_at}\n\n")
        handle.write("| asset | lifecycle | mark | spread | depth 10bps | funding | gate | next step |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.asset} | {row.lifecycle_status} | {row.current_mark} | {row.spread_bps} | "
                f"{row.near_depth_10bps_notional} | {row.annualized_funding} | {row.gate_action} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _gate_for_row(*, row: dict[str, str], context: dict[str, str]) -> OfiCooldownStateGate:
    lifecycle = row.get("lifecycle_status", "")
    second_bps = _float(row.get("second_repeat_5m_bps"))
    action, reason, next_step = _gate_action(lifecycle_status=lifecycle, second_repeat_bps=second_bps)
    return OfiCooldownStateGate(
        asset=row.get("asset", ""),
        decision=row.get("decision", ""),
        lifecycle_status=lifecycle,
        second_repeat_5m_bps=row.get("second_repeat_5m_bps", ""),
        current_mark=_format_float(context.get("mark_price")),
        spread_bps=_format_float(context.get("spread_bps")),
        near_depth_10bps_notional=_format_float(context.get("near_depth_10bps_notional")),
        annualized_funding=_format_float(context.get("annualized_funding")),
        gate_action=action,
        reason=reason,
        next_step=next_step,
    )


def _gate_action(*, lifecycle_status: str, second_repeat_bps: float) -> tuple[str, str, str]:
    if lifecycle_status == "repeat_fill_audit_decay":
        return (
            "block_ofi_repeat",
            "the later fill-audit window rejected the OFI short",
            "wait for a new independent OFI state and do not reuse the existing paper/repeat chain",
        )
    if lifecycle_status == "second_repeat_decay" and second_repeat_bps < 0.0:
        return (
            "cooldown_until_fresh_state",
            "the immediate second repeat lost after earlier OFI wins",
            "require a fresh OFI imbalance state plus a cooldown before another paper short",
        )
    if lifecycle_status == "survives_current_lifecycle":
        return (
            "allow_state_gated_probe",
            "the current lifecycle has not decayed yet",
            "open only one state-gated repeat with stop/adverse-excursion notes",
        )
    return (
        "collect_more_state",
        "lifecycle evidence is incomplete or mixed",
        "collect fresh OFI state before opening more repeats",
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _format_float(value: str | None) -> str:
    if value in (None, ""):
        return ""
    return f"{_float(value):.8f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-decay-path", type=Path, default=ROOT / "current_ofi_timing_decay_review.csv")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_ofi_cooldown_state_gate.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_ofi_cooldown_state_gate.md",
    )
    args = parser.parse_args()

    rows = build_ofi_cooldown_state_gate(timing_decay_path=args.timing_decay_path)
    write_ofi_cooldown_state_gate_csv(rows, output_path=args.output_path)
    write_ofi_cooldown_state_gate_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.asset, row.gate_action, row.next_step)


if __name__ == "__main__":
    main()
