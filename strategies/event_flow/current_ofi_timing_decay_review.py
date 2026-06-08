from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OfiTimingDecayReview:
    asset: str
    decision: str
    initial_paper_bps: str
    initial_fill_audit_15m_bps: str
    first_repeat_5m_bps: str
    repeat_fill_audit_15m_bps: str
    second_repeat_5m_bps: str
    lifecycle_status: str
    interpretation: str
    next_step: str


def build_ofi_timing_decay_review(
    *,
    paper_outcomes_path: Path = ROOT / "current_ofi_paper_outcomes.csv",
    first_fill_audit_outcomes_path: Path = ROOT / "current_ofi_fill_audit_outcomes.csv",
    repeat_outcomes_path: Path = ROOT / "current_ofi_repeat_outcomes.csv",
    repeat_fill_audit_outcomes_path: Path = ROOT / "current_ofi_repeat_fill_audit_outcomes.csv",
    second_repeat_outcomes_path: Path = ROOT / "current_ofi_second_repeat_outcomes.csv",
) -> tuple[OfiTimingDecayReview, ...]:
    paper = _best_by_asset(_read_rows(paper_outcomes_path), value_key="directional_return_bps")
    first_fill = _ready_horizon_by_asset(_read_rows(first_fill_audit_outcomes_path), horizon="15m")
    repeat = _best_by_asset(_read_rows(repeat_outcomes_path), value_key="directional_return_bps")
    repeat_fill = _ready_horizon_by_asset(_read_rows(repeat_fill_audit_outcomes_path), horizon="15m")
    second_repeat = _best_by_asset(_read_rows(second_repeat_outcomes_path), value_key="directional_return_bps")
    assets = sorted(set(paper) | set(first_fill) | set(repeat) | set(repeat_fill) | set(second_repeat))
    return tuple(
        _review_for_asset(
            asset=asset,
            paper=paper.get(asset, {}),
            first_fill=first_fill.get(asset, {}),
            repeat=repeat.get(asset, {}),
            repeat_fill=repeat_fill.get(asset, {}),
            second_repeat=second_repeat.get(asset, {}),
        )
        for asset in assets
    )


def write_ofi_timing_decay_review_csv(rows: tuple[OfiTimingDecayReview, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(tuple(OfiTimingDecayReview.__dataclass_fields__))
        for row in rows:
            writer.writerow(tuple(row.__dict__.values()))
    return output_path


def write_ofi_timing_decay_review_md(rows: tuple[OfiTimingDecayReview, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OFI Timing Decay Review\n\n")
        handle.write(
            "This reviews the OFI short-horizon lifecycle as a timing/state problem. "
            "It is not a trading rule and not a promotion list.\n\n"
        )
        handle.write(
            "| asset | decision | paper | first audit 15m | repeat 5m | repeat audit 15m | second repeat 5m | status | next step |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.asset} | {row.decision} | {row.initial_paper_bps} | "
                f"{row.initial_fill_audit_15m_bps} | {row.first_repeat_5m_bps} | "
                f"{row.repeat_fill_audit_15m_bps} | {row.second_repeat_5m_bps} | "
                f"{row.lifecycle_status} | {_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        for row in rows:
            handle.write(f"- {row.asset}: {row.interpretation}\n")
    return output_path


def _review_for_asset(
    *,
    asset: str,
    paper: dict[str, str],
    first_fill: dict[str, str],
    repeat: dict[str, str],
    repeat_fill: dict[str, str],
    second_repeat: dict[str, str],
) -> OfiTimingDecayReview:
    paper_bps = _float(paper.get("directional_return_bps"))
    first_fill_bps = _float(first_fill.get("close_return_bps"))
    repeat_bps = _float(repeat.get("directional_return_bps"))
    repeat_fill_bps = _float(repeat_fill.get("close_return_bps"))
    second_bps = _float(second_repeat.get("directional_return_bps"))
    decision = paper.get("decision") or first_fill.get("decision") or repeat.get("decision") or repeat_fill.get("decision")
    status, interpretation, next_step = _lifecycle_decision(
        paper_bps=paper_bps,
        first_fill_bps=first_fill_bps,
        repeat_bps=repeat_bps,
        repeat_fill_bps=repeat_fill_bps,
        second_bps=second_bps,
        has_second=bool(second_repeat),
    )
    return OfiTimingDecayReview(
        asset=asset,
        decision=decision,
        initial_paper_bps=_format_optional(paper.get("directional_return_bps")),
        initial_fill_audit_15m_bps=_format_optional(first_fill.get("close_return_bps")),
        first_repeat_5m_bps=_format_optional(repeat.get("directional_return_bps")),
        repeat_fill_audit_15m_bps=_format_optional(repeat_fill.get("close_return_bps")),
        second_repeat_5m_bps=_format_optional(second_repeat.get("directional_return_bps")),
        lifecycle_status=status,
        interpretation=interpretation,
        next_step=next_step,
    )


def _lifecycle_decision(
    *,
    paper_bps: float,
    first_fill_bps: float,
    repeat_bps: float,
    repeat_fill_bps: float,
    second_bps: float,
    has_second: bool,
) -> tuple[str, str, str]:
    if repeat_fill_bps < 0.0:
        return (
            "repeat_fill_audit_decay",
            "The first mark wins did not survive the later fill-audit window.",
            "do not promote; require a fresh independent OFI state before any new label",
        )
    if has_second and second_bps < 0.0:
        return (
            "second_repeat_decay",
            "The signal survived paper, repeat, and fill audit, then failed when immediately chased again.",
            "treat OFI as timing-sensitive; learn entry cooldown/state filters before opening another repeat",
        )
    if paper_bps > 0.0 and first_fill_bps > 0.0 and repeat_bps > 0.0 and repeat_fill_bps > 0.0:
        return (
            "survives_current_lifecycle",
            "The OFI short survived the observed paper, repeat, and fill-audit stages.",
            "open only a state-gated repeat with explicit cooldown and stop/adverse-excursion notes",
        )
    return (
        "incomplete_or_mixed_lifecycle",
        "The OFI lifecycle is incomplete or mixed across stages.",
        "collect more state-aligned labels before using it as a policy preference",
    )


def _best_by_asset(rows: tuple[dict[str, str], ...], *, value_key: str) -> dict[str, dict[str, str]]:
    ready = (row for row in rows if row.get("checkpoint_status") == "ready")
    by_asset: dict[str, dict[str, str]] = {}
    for row in ready:
        asset = row.get("asset", "")
        if not asset:
            continue
        current = by_asset.get(asset)
        if current is None or _float(row.get(value_key)) > _float(current.get(value_key)):
            by_asset[asset] = row
    return by_asset


def _ready_horizon_by_asset(rows: tuple[dict[str, str], ...], *, horizon: str) -> dict[str, dict[str, str]]:
    by_asset = {}
    for row in rows:
        asset = row.get("asset", "")
        if asset and row.get("horizon") == horizon and row.get("checkpoint_status") == "ready":
            by_asset[asset] = row
    return by_asset


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


def _format_optional(value: str | None) -> str:
    if value in (None, ""):
        return ""
    return f"{_float(value):.8f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_ofi_timing_decay_review.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_ofi_timing_decay_review.md")
    args = parser.parse_args()

    rows = build_ofi_timing_decay_review()
    write_ofi_timing_decay_review_csv(rows, output_path=args.output_path)
    write_ofi_timing_decay_review_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.asset, row.lifecycle_status, row.next_step)


if __name__ == "__main__":
    main()
