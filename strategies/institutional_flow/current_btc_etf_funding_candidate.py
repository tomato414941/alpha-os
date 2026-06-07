from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CurrentBtcEtfFundingCandidate:
    asset: str
    status: str
    side: str
    latest_date: str
    latest_flow_btc: float
    rolling_5d_flow_btc: float
    annualized_funding: float
    open_interest_notional: float
    day_notional_volume: float
    historical_trades: int
    historical_total_return: float
    historical_hit_rate_5d: float
    historical_max_drawdown: float
    reason: str


def build_current_candidate(
    *,
    current_join_path: Path,
    historical_summary_path: Path,
) -> CurrentBtcEtfFundingCandidate | None:
    current_rows = _read_rows(current_join_path)
    summary_rows = _read_rows(historical_summary_path)
    if not current_rows or not summary_rows:
        return None
    current = current_rows[0]
    summary = summary_rows[0]
    rolling_5d = _float(current["rolling_5d_flow_btc"])
    funding = _float(current["annualized_funding"])
    active = rolling_5d <= -15_000.0 and funding > 0.0
    return CurrentBtcEtfFundingCandidate(
        asset=current["asset"],
        status="active_paper_watch" if active else "inactive",
        side="short_btc_perp" if active else "none",
        latest_date=current["latest_date"],
        latest_flow_btc=_float(current["latest_flow_btc"]),
        rolling_5d_flow_btc=rolling_5d,
        annualized_funding=funding,
        open_interest_notional=_float(current["open_interest_notional"]),
        day_notional_volume=_float(current["day_notional_volume"]),
        historical_trades=int(summary["trades"]),
        historical_total_return=_float(summary["total_return"]),
        historical_hit_rate_5d=_float(summary["hit_rate_5d"]),
        historical_max_drawdown=_float(summary["max_drawdown"]),
        reason=_reason(active=active, rolling_5d_flow_btc=rolling_5d, annualized_funding=funding),
    )


def write_candidate_csv(
    candidate: CurrentBtcEtfFundingCandidate | None,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "status",
                "side",
                "latest_date",
                "latest_flow_btc",
                "rolling_5d_flow_btc",
                "annualized_funding",
                "open_interest_notional",
                "day_notional_volume",
                "historical_trades",
                "historical_total_return",
                "historical_hit_rate_5d",
                "historical_max_drawdown",
                "reason",
            )
        )
        if candidate is not None:
            writer.writerow(
                (
                    candidate.asset,
                    candidate.status,
                    candidate.side,
                    candidate.latest_date,
                    f"{candidate.latest_flow_btc:.8f}",
                    f"{candidate.rolling_5d_flow_btc:.8f}",
                    f"{candidate.annualized_funding:.8f}",
                    f"{candidate.open_interest_notional:.8f}",
                    f"{candidate.day_notional_volume:.8f}",
                    candidate.historical_trades,
                    f"{candidate.historical_total_return:.8f}",
                    f"{candidate.historical_hit_rate_5d:.8f}",
                    f"{candidate.historical_max_drawdown:.8f}",
                    candidate.reason,
                )
            )
    return output_path


def write_candidate_md(
    candidate: CurrentBtcEtfFundingCandidate | None,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current BTC ETF Funding Candidate\n\n")
        handle.write(
            "This is a current paper watch for the BTC ETF-flow/funding rule. It is not a live trade instruction.\n\n"
        )
        handle.write(
            "| asset | status | side | latest date | latest flow BTC | 5d flow BTC | funding ann | hist trades | hist total | hist hit | hist mdd | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        if candidate is not None:
            handle.write(
                f"| {candidate.asset} | {candidate.status} | {candidate.side} | "
                f"{candidate.latest_date} | {candidate.latest_flow_btc:.2f} | "
                f"{candidate.rolling_5d_flow_btc:.2f} | {candidate.annualized_funding:.8f} | "
                f"{candidate.historical_trades} | {candidate.historical_total_return:.8f} | "
                f"{candidate.historical_hit_rate_5d:.4f} | {candidate.historical_max_drawdown:.8f} | "
                f"{candidate.reason} |\n"
            )
        handle.write("\n## Caveat\n\n")
        handle.write(
            "A current watch still needs intraday entry, liquidation buffer, actual fees, and exchange-specific execution checks before any real trade.\n"
        )
    return output_path


def _reason(*, active: bool, rolling_5d_flow_btc: float, annualized_funding: float) -> str:
    if active:
        return "large rolling ETF outflow and positive perp funding align with historical short BTC candidate"
    if rolling_5d_flow_btc > -15_000.0:
        return "rolling ETF outflow is not large enough"
    if annualized_funding <= 0.0:
        return "BTC perp funding does not pay shorts"
    return "rule conditions are not met"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-join-path",
        type=Path,
        default=ROOT / "current_btc_etf_market_join.csv",
    )
    parser.add_argument(
        "--historical-summary-path",
        type=Path,
        default=ROOT / "btc_etf_flow_funding_candidate_summary.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_btc_etf_funding_candidate.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_btc_etf_funding_candidate.md",
    )
    args = parser.parse_args()

    candidate = build_current_candidate(
        current_join_path=args.current_join_path,
        historical_summary_path=args.historical_summary_path,
    )
    write_candidate_csv(candidate, output_path=args.output_path)
    write_candidate_md(candidate, output_path=args.markdown_output_path)
    if candidate is not None:
        print(
            candidate.asset,
            candidate.status,
            candidate.side,
            f"flow5d={candidate.rolling_5d_flow_btc:.2f}",
            f"funding={candidate.annualized_funding:.6f}",
            candidate.reason,
        )


if __name__ == "__main__":
    main()
