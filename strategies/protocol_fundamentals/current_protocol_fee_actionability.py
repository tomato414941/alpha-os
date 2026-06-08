from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProtocolFeeActionabilityRow:
    token_symbol: str
    protocol: str
    side: str
    status: str
    action: str
    score: float
    thesis_status: str
    thesis_score: float
    fee_to_market_cap: float
    fee_growth_7d: float
    price_change_7d: float
    execution_action: str
    venue_count: int
    spread_bps: float
    depth_10bps: float
    label_observations: int
    labeled_4h: int
    wins_4h: int
    mean_directional_4h: float
    latest_label_status: str
    reason: str
    next_step: str


def build_protocol_fee_actionability_rows(root: Path = ROOT) -> tuple[ProtocolFeeActionabilityRow, ...]:
    executions = {
        (row.get("token_symbol", ""), row.get("protocol", "")): row
        for row in _read_rows(root / "current_protocol_fee_execution_context.csv")
        if row.get("token_symbol") and row.get("protocol")
    }
    labels_by_key = _labels_by_key(root / "current_protocol_fee_price_lag_labels.csv")

    output: list[ProtocolFeeActionabilityRow] = []
    for row in _read_rows(root / "current_protocol_fee_price_context.csv"):
        if row.get("status") not in {
            "fee_growth_price_lag_candidate",
            "fee_growth_price_confirmation",
            "fee_growth_price_chase_risk",
            "fee_decay_price_weakness_context",
        }:
            continue
        key = (row.get("token_symbol", ""), row.get("protocol", ""))
        execution = executions.get(key, {})
        labels = labels_by_key.get(key, ())
        labeled_4h = tuple(label for label in labels if label.get("directional_return_4h"))
        directional_4h = tuple(_float(label.get("directional_return_4h")) for label in labeled_4h)
        wins_4h = sum(1 for value in directional_4h if value > 0.0)
        mean_directional_4h = sum(directional_4h) / len(directional_4h) if directional_4h else 0.0
        latest_label_status = labels[0].get("label_status", "") if labels else ""
        status, action, reason = _status_action_reason(
            execution_action=execution.get("action", ""),
            label_observations=len(labels),
            labeled_4h=len(labeled_4h),
            wins_4h=wins_4h,
            mean_directional_4h=mean_directional_4h,
            latest_label_status=latest_label_status,
        )
        token = row.get("token_symbol", "")
        output.append(
            ProtocolFeeActionabilityRow(
                token_symbol=token,
                protocol=row.get("protocol", ""),
                side=row.get("side", ""),
                status=status,
                action=action,
                score=_score(
                    status=status,
                    thesis_score=_float(row.get("score")),
                    execution_action=execution.get("action", ""),
                    labeled_4h=len(labeled_4h),
                    wins_4h=wins_4h,
                    mean_directional_4h=mean_directional_4h,
                    spread_bps=_float(execution.get("hl_spread_bps")),
                ),
                thesis_status=row.get("status", ""),
                thesis_score=_float(row.get("score")),
                fee_to_market_cap=_float(row.get("fee_to_market_cap")),
                fee_growth_7d=_float(row.get("fee_growth_7d")),
                price_change_7d=_float(row.get("price_change_7d")),
                execution_action=execution.get("action", ""),
                venue_count=_int(execution.get("venue_count")),
                spread_bps=_float(execution.get("hl_spread_bps")),
                depth_10bps=_float(execution.get("hl_near_depth_10bps_notional")),
                label_observations=len(labels),
                labeled_4h=len(labeled_4h),
                wins_4h=wins_4h,
                mean_directional_4h=mean_directional_4h,
                latest_label_status=latest_label_status,
                reason=reason,
                next_step=_next_step(token=token, status=status),
            )
        )
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_protocol_fee_actionability_csv(
    rows: tuple[ProtocolFeeActionabilityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "token_symbol",
                "protocol",
                "side",
                "status",
                "action",
                "score",
                "thesis_status",
                "thesis_score",
                "fee_to_market_cap",
                "fee_growth_7d",
                "price_change_7d",
                "execution_action",
                "venue_count",
                "spread_bps",
                "depth_10bps",
                "label_observations",
                "labeled_4h",
                "wins_4h",
                "mean_directional_4h",
                "latest_label_status",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.token_symbol,
                    row.protocol,
                    row.side,
                    row.status,
                    row.action,
                    f"{row.score:.8f}",
                    row.thesis_status,
                    f"{row.thesis_score:.8f}",
                    f"{row.fee_to_market_cap:.8f}",
                    f"{row.fee_growth_7d:.8f}",
                    f"{row.price_change_7d:.8f}",
                    row.execution_action,
                    row.venue_count,
                    f"{row.spread_bps:.8f}",
                    f"{row.depth_10bps:.8f}",
                    row.label_observations,
                    row.labeled_4h,
                    row.wins_4h,
                    f"{row.mean_directional_4h:.8f}",
                    row.latest_label_status,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_protocol_fee_actionability_md(
    rows: tuple[ProtocolFeeActionabilityRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Fee Actionability\n\n")
        handle.write(
            "This separates fee-growth price-context screens from candidates with mature forward labels. "
            "Protocol fees are not assumed to be token-holder revenue.\n\n"
        )
        handle.write(
            "| token | protocol | status | action | score | thesis | exec | labels | 4h wins | mean 4h | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.token_symbol} | {row.protocol} | {row.status} | {row.action} | {row.score:.4f} | "
                f"{row.thesis_status} {row.thesis_score:.2f} | {row.execution_action} | "
                f"{row.label_observations} | {row.wins_4h} | {row.mean_directional_4h:.6f} | "
                f"{_escape(row.reason)} |\n"
            )
    return output_path


def _labels_by_key(path: Path) -> dict[tuple[str, str], tuple[dict[str, str], ...]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _read_rows(path):
        key = (row.get("token_symbol", ""), row.get("protocol", ""))
        if all(key):
            grouped.setdefault(key, []).append(row)
    return {key: tuple(sorted(rows, key=lambda row: row.get("observed_at", ""), reverse=True)) for key, rows in grouped.items()}


def _status_action_reason(
    *,
    execution_action: str,
    label_observations: int,
    labeled_4h: int,
    wins_4h: int,
    mean_directional_4h: float,
    latest_label_status: str,
) -> tuple[str, str, str]:
    if labeled_4h >= 2 and wins_4h >= 2 and mean_directional_4h > 0.0 and execution_action == "paper_observation_ready":
        return (
            "fee_growth_repeat_execution_candidate",
            "repeat_paper_probe",
            "repeated 4h labels and current public-book context both pass",
        )
    if labeled_4h >= 1 and wins_4h >= 1 and mean_directional_4h > 0.0:
        return (
            "fee_growth_label_supported_watch",
            "refresh_execution_gate",
            "at least one 4h label supports the fee-growth direction but execution or repetition is not enough",
        )
    if labeled_4h >= 1:
        return (
            "fee_growth_label_failed",
            "deprioritize_until_fresh_snapshot",
            "mature 4h labels do not support the fee-growth direction",
        )
    if label_observations > 0:
        return (
            "fee_growth_pending_forward_label",
            "wait_for_forward_label",
            f"forward label is not mature yet: {latest_label_status}",
        )
    return (
        "fee_growth_unlabeled_watch",
        "create_forward_label",
        "fee-growth screen has no stored forward-label observation",
    )


def _score(
    *,
    status: str,
    thesis_score: float,
    execution_action: str,
    labeled_4h: int,
    wins_4h: int,
    mean_directional_4h: float,
    spread_bps: float,
) -> float:
    status_base = {
        "fee_growth_repeat_execution_candidate": 80.0,
        "fee_growth_label_supported_watch": 62.0,
        "fee_growth_pending_forward_label": 46.0,
        "fee_growth_unlabeled_watch": 36.0,
        "fee_growth_label_failed": 20.0,
    }.get(status, 0.0)
    execution_bonus = {
        "paper_observation_ready": 6.0,
        "non_hyperliquid_route_check": 2.0,
        "wide_spread_watch": -3.0,
        "thin_volume_watch": -5.0,
        "thin_depth_watch": -5.0,
        "venue_gap": -12.0,
    }.get(execution_action, 0.0)
    label_bonus = wins_4h * 6.0 + max(mean_directional_4h, -0.05) * 200.0
    spread_penalty = min(max(spread_bps - 5.0, 0.0), 20.0) * 0.3
    return status_base + min(thesis_score, 100.0) / 25.0 + labeled_4h * 2.0 + label_bonus + execution_bonus - spread_penalty


def _next_step(*, token: str, status: str) -> str:
    if status == "fee_growth_repeat_execution_candidate":
        return f"repeat {token} fee-growth paper probe with 4h/12h/24h labels and execution costs"
    if status == "fee_growth_label_supported_watch":
        return f"refresh {token} execution gate and require another positive 4h label before promotion"
    if status == "fee_growth_pending_forward_label":
        return f"wait for {token} 4h forward label before treating this as an alpha candidate"
    if status == "fee_growth_label_failed":
        return f"deprioritize {token} until a fresh fee-growth snapshot appears"
    return f"store a forward-label observation for {token}"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _int(value: str | None) -> int:
    return int(float(value)) if value else 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_protocol_fee_actionability.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_actionability.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_protocol_fee_actionability_rows()
    write_protocol_fee_actionability_csv(rows, output_path=args.output_path)
    write_protocol_fee_actionability_md(rows, output_path=args.markdown_output_path, top=args.top)

    if rows:
        best = rows[0]
        print(
            "best_protocol_fee_actionability",
            best.token_symbol,
            best.status,
            f"score={best.score:.4f}",
            f"labels={best.label_observations}",
        )


if __name__ == "__main__":
    main()
