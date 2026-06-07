from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from btc_etf_flow_funding_regime_summary import ROOT, build_funding_enriched_label_rows


@dataclass(frozen=True)
class BtcEtfFlowFundingTrade:
    entry_date: str
    exit_date: str
    flow_date: str
    flow_btc: float
    rolling_5d_flow_btc: float
    start_funding_support: float
    directional_return_5d: float
    funding_support_5d: float
    round_trip_fee: float
    net_return_5d: float
    equity_after: float
    drawdown_after: float


@dataclass(frozen=True)
class BtcEtfFlowFundingRuleSummary:
    rule_key: str
    trades: int
    skipped_overlap_signals: int
    total_return: float
    mean_net_return_5d: float
    hit_rate_5d: float
    max_drawdown: float
    fee_bps_per_side: float
    action: str


@dataclass(frozen=True)
class BtcEtfFlowFundingRobustnessRow:
    group_key: str
    trades: int
    total_return: float
    mean_net_return_5d: float
    hit_rate_5d: float
    max_drawdown: float
    action: str


def build_candidate_trades(
    *,
    labels_path: Path,
    fee_bps_per_side: float,
    max_workers: int = 12,
) -> tuple[BtcEtfFlowFundingTrade, ...]:
    rows = tuple(
        row
        for row in build_funding_enriched_label_rows(
            labels_path=labels_path,
            max_workers=max_workers,
        )
        if _is_large_outflow_start_funding_signal(row)
    )
    return build_candidate_trades_from_rows(rows, fee_bps_per_side=fee_bps_per_side)


def build_candidate_trades_from_rows(
    rows: tuple[dict[str, str], ...],
    *,
    fee_bps_per_side: float,
) -> tuple[BtcEtfFlowFundingTrade, ...]:
    trades: list[BtcEtfFlowFundingTrade] = []
    equity = 1.0
    peak_equity = 1.0
    next_available_entry = date.min
    round_trip_fee = (fee_bps_per_side * 2.0) / 10_000.0
    for row in sorted(rows, key=lambda item: item["label_start_date"]):
        entry_date = date.fromisoformat(row["label_start_date"])
        if entry_date < next_available_entry:
            continue
        exit_date = entry_date + timedelta(days=5)
        directional_return = float(row["directional_return_5d"])
        funding_support = float(row["funding_support_5d"])
        net_return = directional_return + funding_support - round_trip_fee
        equity *= 1.0 + net_return
        peak_equity = max(peak_equity, equity)
        drawdown = (equity / peak_equity) - 1.0 if peak_equity > 0.0 else 0.0
        trades.append(
            BtcEtfFlowFundingTrade(
                entry_date=entry_date.isoformat(),
                exit_date=exit_date.isoformat(),
                flow_date=row["flow_date"],
                flow_btc=float(row["flow_btc"]),
                rolling_5d_flow_btc=float(row["rolling_5d_flow_btc"]),
                start_funding_support=float(row["start_funding_support"]),
                directional_return_5d=directional_return,
                funding_support_5d=funding_support,
                round_trip_fee=round_trip_fee,
                net_return_5d=net_return,
                equity_after=equity,
                drawdown_after=drawdown,
            )
        )
        next_available_entry = exit_date
    return tuple(trades)


def build_fee_sensitivity_rows(
    *,
    signal_rows: tuple[dict[str, str], ...],
    fee_bps_values: tuple[float, ...],
) -> tuple[BtcEtfFlowFundingRobustnessRow, ...]:
    return tuple(
        _robustness_row(
            group_key=f"fee_bps_per_side_{fee_bps:.1f}",
            trades=build_candidate_trades_from_rows(
                signal_rows,
                fee_bps_per_side=fee_bps,
            ),
        )
        for fee_bps in fee_bps_values
    )


def build_year_split_rows(
    trades: tuple[BtcEtfFlowFundingTrade, ...],
) -> tuple[BtcEtfFlowFundingRobustnessRow, ...]:
    years = tuple(sorted({trade.entry_date[:4] for trade in trades}))
    return tuple(
        _robustness_row(
            group_key=f"entry_year_{year}",
            trades=tuple(trade for trade in trades if trade.entry_date[:4] == year),
        )
        for year in years
    )


def summarize_candidate_rule(
    trades: tuple[BtcEtfFlowFundingTrade, ...],
    *,
    total_signal_count: int,
    fee_bps_per_side: float,
) -> BtcEtfFlowFundingRuleSummary:
    net_returns = tuple(trade.net_return_5d for trade in trades)
    total_return = trades[-1].equity_after - 1.0 if trades else 0.0
    max_drawdown = min((trade.drawdown_after for trade in trades), default=0.0)
    summary = BtcEtfFlowFundingRuleSummary(
        rule_key="large_5d_outflow_start_funding_aligned_5d_hold",
        trades=len(trades),
        skipped_overlap_signals=max(0, total_signal_count - len(trades)),
        total_return=total_return,
        mean_net_return_5d=_mean(net_returns),
        hit_rate_5d=_hit_rate(net_returns),
        max_drawdown=max_drawdown,
        fee_bps_per_side=fee_bps_per_side,
        action="",
    )
    return BtcEtfFlowFundingRuleSummary(
        **{
            **summary.__dict__,
            "action": _action_for_summary(summary),
        }
    )


def write_robustness_csv(
    rows: tuple[BtcEtfFlowFundingRobustnessRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "group_key",
                "trades",
                "total_return",
                "mean_net_return_5d",
                "hit_rate_5d",
                "max_drawdown",
                "action",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.group_key,
                    row.trades,
                    f"{row.total_return:.8f}",
                    f"{row.mean_net_return_5d:.8f}",
                    f"{row.hit_rate_5d:.8f}",
                    f"{row.max_drawdown:.8f}",
                    row.action,
                )
            )
    return output_path


def write_trades_csv(
    trades: tuple[BtcEtfFlowFundingTrade, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "entry_date",
                "exit_date",
                "flow_date",
                "flow_btc",
                "rolling_5d_flow_btc",
                "start_funding_support",
                "directional_return_5d",
                "funding_support_5d",
                "round_trip_fee",
                "net_return_5d",
                "equity_after",
                "drawdown_after",
            )
        )
        for trade in trades:
            writer.writerow(
                (
                    trade.entry_date,
                    trade.exit_date,
                    trade.flow_date,
                    f"{trade.flow_btc:.8f}",
                    f"{trade.rolling_5d_flow_btc:.8f}",
                    f"{trade.start_funding_support:.12f}",
                    f"{trade.directional_return_5d:.8f}",
                    f"{trade.funding_support_5d:.12f}",
                    f"{trade.round_trip_fee:.8f}",
                    f"{trade.net_return_5d:.8f}",
                    f"{trade.equity_after:.8f}",
                    f"{trade.drawdown_after:.8f}",
                )
            )
    return output_path


def write_summary_csv(
    summary: BtcEtfFlowFundingRuleSummary,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "rule_key",
                "trades",
                "skipped_overlap_signals",
                "total_return",
                "mean_net_return_5d",
                "hit_rate_5d",
                "max_drawdown",
                "fee_bps_per_side",
                "action",
            )
        )
        writer.writerow(
            (
                summary.rule_key,
                summary.trades,
                summary.skipped_overlap_signals,
                f"{summary.total_return:.8f}",
                f"{summary.mean_net_return_5d:.8f}",
                f"{summary.hit_rate_5d:.8f}",
                f"{summary.max_drawdown:.8f}",
                f"{summary.fee_bps_per_side:.4f}",
                summary.action,
            )
        )
    return output_path


def write_summary_md(
    summary: BtcEtfFlowFundingRuleSummary,
    *,
    robustness_rows: tuple[BtcEtfFlowFundingRobustnessRow, ...],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# BTC ETF Flow Funding Candidate Rule\n\n")
        handle.write(
            "This is a non-overlapping paper rule for the large rolling ETF outflow plus start-funding-aligned BTC short candidate. "
            "It uses only label-start funding for entry filtering, then adds observed 5-day funding as rough PnL.\n\n"
        )
        handle.write("| rule | trades | skipped | total return | mean net 5d | hit 5d | max drawdown | fee bps/side | action |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        handle.write(
            f"| {summary.rule_key} | {summary.trades} | {summary.skipped_overlap_signals} | "
            f"{summary.total_return:.8f} | {summary.mean_net_return_5d:.8f} | "
            f"{summary.hit_rate_5d:.4f} | {summary.max_drawdown:.8f} | "
            f"{summary.fee_bps_per_side:.4f} | {summary.action} |\n\n"
        )
        handle.write("## Robustness\n\n")
        handle.write("| group | trades | total return | mean net 5d | hit 5d | max drawdown | action |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in robustness_rows:
            handle.write(
                f"| {row.group_key} | {row.trades} | {row.total_return:.8f} | "
                f"{row.mean_net_return_5d:.8f} | {row.hit_rate_5d:.4f} | "
                f"{row.max_drawdown:.8f} | {row.action} |\n"
            )
        handle.write("\n")
        handle.write("## Caveat\n\n")
        handle.write(
            "This is not live-ready. It still ignores intraday fill timing, mark/index basis, liquidation buffer, and account-specific fees. "
            "Its value is that it removes overlapping signal inflation and stops using future funding as an entry condition.\n"
        )
    return output_path


def _is_large_outflow_start_funding_signal(row: dict[str, str]) -> bool:
    return (
        int(row["direction_hint"]) == -1
        and float(row["rolling_5d_flow_btc"]) <= -15_000.0
        and float(row["start_funding_support"]) > 0.0
    )


def _action_for_summary(summary: BtcEtfFlowFundingRuleSummary) -> str:
    if summary.trades >= 10 and summary.total_return > 0.0 and summary.hit_rate_5d >= 0.55:
        return "paper_rule_candidate"
    if summary.trades >= 5 and summary.total_return > 0.0:
        return "paper_rule_watch"
    return "weak_or_insufficient"


def _robustness_row(
    *,
    group_key: str,
    trades: tuple[BtcEtfFlowFundingTrade, ...],
) -> BtcEtfFlowFundingRobustnessRow:
    net_returns = tuple(trade.net_return_5d for trade in trades)
    total_return = _compounded_return(net_returns)
    row = BtcEtfFlowFundingRobustnessRow(
        group_key=group_key,
        trades=len(trades),
        total_return=total_return,
        mean_net_return_5d=_mean(net_returns),
        hit_rate_5d=_hit_rate(net_returns),
        max_drawdown=_max_drawdown(net_returns),
        action="",
    )
    return BtcEtfFlowFundingRobustnessRow(
        **{
            **row.__dict__,
            "action": _action_for_robustness_row(row),
        }
    )


def _action_for_robustness_row(row: BtcEtfFlowFundingRobustnessRow) -> str:
    if row.trades >= 5 and row.total_return > 0.0 and row.hit_rate_5d >= 0.55:
        return "survives"
    if row.trades >= 3 and row.total_return > 0.0:
        return "thin_positive"
    return "fails_or_too_thin"


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _hit_rate(values: tuple[float, ...]) -> float:
    return sum(1.0 for value in values if value > 0.0) / len(values) if values else 0.0


def _compounded_return(values: tuple[float, ...]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + value
    return equity - 1.0


def _max_drawdown(values: tuple[float, ...]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, (equity / peak) - 1.0 if peak > 0.0 else 0.0)
    return max_drawdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=ROOT / "btc_etf_flow_forward_labels.csv",
    )
    parser.add_argument(
        "--trades-output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_funding_candidate_trades.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_funding_candidate_summary.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_funding_candidate_summary.md",
    )
    parser.add_argument(
        "--robustness-output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_funding_candidate_robustness.csv",
    )
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()

    enriched_rows = build_funding_enriched_label_rows(
        labels_path=args.labels_path,
        max_workers=args.max_workers,
    )
    signal_rows = tuple(row for row in enriched_rows if _is_large_outflow_start_funding_signal(row))
    trades = build_candidate_trades_from_rows(
        signal_rows,
        fee_bps_per_side=args.fee_bps_per_side,
    )
    summary = summarize_candidate_rule(
        trades,
        total_signal_count=len(signal_rows),
        fee_bps_per_side=args.fee_bps_per_side,
    )
    robustness_rows = (
        *build_fee_sensitivity_rows(
            signal_rows=signal_rows,
            fee_bps_values=(1.0, 5.0, 10.0, 20.0, 50.0),
        ),
        *build_year_split_rows(trades),
    )
    write_trades_csv(trades, output_path=args.trades_output_path)
    write_summary_csv(summary, output_path=args.summary_output_path)
    write_robustness_csv(robustness_rows, output_path=args.robustness_output_path)
    write_summary_md(
        summary,
        robustness_rows=robustness_rows,
        output_path=args.markdown_output_path,
    )
    print(
        summary.rule_key,
        f"trades={summary.trades}",
        f"total={summary.total_return:.8f}",
        f"mean={summary.mean_net_return_5d:.8f}",
        f"hit={summary.hit_rate_5d:.4f}",
        f"mdd={summary.max_drawdown:.8f}",
        summary.action,
    )


if __name__ == "__main__":
    main()
