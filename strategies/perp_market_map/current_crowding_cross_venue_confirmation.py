from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class CrossVenueCrowdingConfirmation:
    asset: str
    action: str
    status: str
    score: float
    label_status: str
    label_observations: int
    net_directional_return_1h_proxy: float
    venue_count: int
    actionable_venue_count: int
    max_derivatives_score: float
    max_oi_volume_ratio: float
    max_abs_funding_rate: float
    venue_examples: str
    reason: str
    next_step: str


def build_cross_venue_confirmations(
    *,
    validated_path: Path = ROOT / "current_crowding_reversion_validated_candidates.csv",
    derivatives_path: Path = ROOT / "current_crowding_derivatives_coverage.csv",
) -> tuple[CrossVenueCrowdingConfirmation, ...]:
    derivatives_by_asset = _derivatives_by_asset(_read_rows(derivatives_path))
    rows = tuple(
        _build_row(validated=row, derivatives=derivatives_by_asset.get(row.get("asset", ""), ()))
        for row in _read_rows(validated_path)
    )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_confirmations_csv(rows: tuple[CrossVenueCrowdingConfirmation, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "action",
                "status",
                "score",
                "label_status",
                "label_observations",
                "net_directional_return_1h_proxy",
                "venue_count",
                "actionable_venue_count",
                "max_derivatives_score",
                "max_oi_volume_ratio",
                "max_abs_funding_rate",
                "venue_examples",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    row.status,
                    f"{row.score:.8f}",
                    row.label_status,
                    row.label_observations,
                    f"{row.net_directional_return_1h_proxy:.8f}",
                    row.venue_count,
                    row.actionable_venue_count,
                    f"{row.max_derivatives_score:.8f}",
                    f"{row.max_oi_volume_ratio:.8f}",
                    f"{row.max_abs_funding_rate:.8f}",
                    row.venue_examples,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_confirmations_md(rows: tuple[CrossVenueCrowdingConfirmation, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crowding Cross-Venue Confirmation\n\n")
        handle.write(
            "This joins Hyperliquid crowding-reversion labels with CoinGecko multi-venue derivatives context. "
            "It checks whether a crowded-positioning idea has visible cross-venue context. "
            "It is not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | status | score | labels | net1h proxy | venues | max OI/vol | max funding | examples | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:25]:
            handle.write(
                f"| {row.asset} | "
                f"{row.action} | "
                f"{row.status} | "
                f"{row.score:.4f} | "
                f"{row.label_observations} | "
                f"{row.net_directional_return_1h_proxy:.6f} | "
                f"{row.venue_count}/{row.actionable_venue_count} | "
                f"{row.max_oi_volume_ratio:.4f} | "
                f"{row.max_abs_funding_rate:.4f} | "
                f"{_escape(row.venue_examples)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _build_row(
    *,
    validated: dict[str, str],
    derivatives: tuple[dict[str, str], ...],
) -> CrossVenueCrowdingConfirmation:
    asset = validated.get("asset", "")
    label_observations = _int(validated.get("label_observations"))
    net_1h = _float(validated.get("net_directional_return_1h_proxy"))
    actionable = tuple(row for row in derivatives if row.get("status") != "derivatives_positioning_context")
    max_derivatives_score = max((_float(row.get("score")) for row in derivatives), default=0.0)
    max_oi_volume_ratio = max((_float(row.get("oi_volume_ratio")) for row in derivatives), default=0.0)
    max_abs_funding_rate = max((abs(_float(row.get("funding_rate"))) for row in derivatives), default=0.0)
    status, reason = _status(
        derivatives=derivatives,
        actionable=actionable,
        label_observations=label_observations,
        net_1h=net_1h,
    )
    score = (
        _float(validated.get("validation_score"))
        + min(len(derivatives), 6) * 1.5
        + min(len(actionable), 4) * 1.0
        + max_derivatives_score * 0.15
        + (5.0 if net_1h > 0.0 else 0.0)
    )
    return CrossVenueCrowdingConfirmation(
        asset=asset,
        action=validated.get("action", ""),
        status=status,
        score=score,
        label_status=validated.get("status", ""),
        label_observations=label_observations,
        net_directional_return_1h_proxy=net_1h,
        venue_count=len(derivatives),
        actionable_venue_count=len(actionable),
        max_derivatives_score=max_derivatives_score,
        max_oi_volume_ratio=max_oi_volume_ratio,
        max_abs_funding_rate=max_abs_funding_rate,
        venue_examples=_venue_examples(derivatives),
        reason=reason,
        next_step=_next_step(asset=asset, status=status),
    )


def _status(
    *,
    derivatives: tuple[dict[str, str], ...],
    actionable: tuple[dict[str, str], ...],
    label_observations: int,
    net_1h: float,
) -> tuple[str, str]:
    if not derivatives:
        return "mapping_gap_no_cross_venue_context", "no matching CoinGecko derivatives rows"
    if net_1h > 0.0 and label_observations >= 6 and len(actionable) >= 2:
        return "cross_venue_label_supported", "label and multi-venue context are both visible"
    if len(actionable) >= 2:
        return "cross_venue_context_ready_needs_labels", "multi-venue context exists but labels are not yet positive"
    if len(derivatives) >= 2:
        return "cross_venue_context_only", "multiple venues exist but not enough actionable venue rows"
    return "single_venue_context_only", "only one matching derivatives venue is visible"


def _next_step(*, asset: str, status: str) -> str:
    if status == "cross_venue_label_supported":
        return f"paper-check {asset} crowding unwind with venue-specific fees, funding, spread, and stop behavior"
    if status == "mapping_gap_no_cross_venue_context":
        return f"fix symbol mapping before using {asset} as a crowding-unwind factor"
    return f"collect more {asset} crowding labels and require cross-venue OI/funding confirmation"


def _derivatives_by_asset(rows: tuple[dict[str, str], ...]) -> dict[str, tuple[dict[str, str], ...]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        asset = _asset_key(row)
        if not asset:
            continue
        grouped.setdefault(asset, []).append(row)
    return {asset: tuple(sorted(values, key=lambda row: _float(row.get("score")), reverse=True)) for asset, values in grouped.items()}


def _asset_key(row: dict[str, str]) -> str:
    index_id = row.get("index_id", "").upper()
    if index_id and index_id not in {"-", "UNKNOWN"}:
        return index_id.removeprefix("K")
    symbol = row.get("symbol", "").upper()
    for suffix in ("USDTM", "USDT", "-PERP", "_PERP", "-USD", "_USD", "PERP"):
        symbol = symbol.replace(suffix, "")
    return symbol.split("/")[0].split("-")[0].split("_")[0]


def _venue_examples(rows: tuple[dict[str, str], ...]) -> str:
    return ";".join(
        f"{row.get('market', '')}/{row.get('symbol', '')}/score={row.get('score', '')}"
        for row in rows[:3]
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _int(value: str | None) -> int:
    if value in {None, ""}:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated-path", type=Path, default=ROOT / "current_crowding_reversion_validated_candidates.csv")
    parser.add_argument(
        "--derivatives-path",
        type=Path,
        default=ROOT / "current_crowding_derivatives_coverage.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_crowding_cross_venue_confirmation.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_crowding_cross_venue_confirmation.md",
    )
    args = parser.parse_args()
    rows = build_cross_venue_confirmations(
        validated_path=args.validated_path,
        derivatives_path=args.derivatives_path,
    )
    write_confirmations_csv(rows, output_path=args.output_path)
    write_confirmations_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(row.asset, row.status, f"{row.score:.4f}")


if __name__ == "__main__":
    main()
