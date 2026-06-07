from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BtcEtfMarketContext:
    asset: str
    etf_action: str
    latest_date: str
    latest_flow_btc: float
    rolling_5d_flow_btc: float
    annualized_funding: float
    open_interest_notional: float
    day_notional_volume: float
    action: str
    score: float


def join_btc_etf_to_market(
    *,
    snapshot_path: Path,
    hyperliquid_path: Path,
) -> tuple[BtcEtfMarketContext, ...]:
    snapshot_rows = _read_rows(snapshot_path)
    market_rows = _read_rows(hyperliquid_path)
    if not snapshot_rows:
        return ()
    snapshot = snapshot_rows[0]
    btc_market = next((row for row in market_rows if row.get("asset") == "BTC"), None)
    if btc_market is None:
        return ()
    funding = _float(btc_market.get("annualized_funding", ""))
    open_interest = _float(btc_market.get("open_interest_notional", ""))
    volume = _float(btc_market.get("day_notional_volume", ""))
    rolling_5d = _float(snapshot["rolling_5d_flow_btc"])
    score = _float(snapshot["score"]) + min(open_interest / max(volume, 1.0), 3.0)
    context = BtcEtfMarketContext(
        asset="BTC",
        etf_action=snapshot["action"],
        latest_date=snapshot["latest_date"],
        latest_flow_btc=_float(snapshot["latest_flow_btc"]),
        rolling_5d_flow_btc=rolling_5d,
        annualized_funding=funding,
        open_interest_notional=open_interest,
        day_notional_volume=volume,
        action=_action_for_context(etf_action=snapshot["action"], funding=funding),
        score=score,
    )
    return (context,)


def write_contexts(
    contexts: tuple[BtcEtfMarketContext, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "etf_action",
                "latest_date",
                "latest_flow_btc",
                "rolling_5d_flow_btc",
                "annualized_funding",
                "open_interest_notional",
                "day_notional_volume",
                "action",
                "score",
            )
        )
        for context in contexts:
            writer.writerow(
                (
                    context.asset,
                    context.etf_action,
                    context.latest_date,
                    f"{context.latest_flow_btc:.8f}",
                    f"{context.rolling_5d_flow_btc:.8f}",
                    f"{context.annualized_funding:.8f}",
                    f"{context.open_interest_notional:.8f}",
                    f"{context.day_notional_volume:.8f}",
                    context.action,
                    f"{context.score:.8f}",
                )
            )
    return output_path


def write_markdown(
    contexts: tuple[BtcEtfMarketContext, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current BTC ETF Market Join\n\n")
        handle.write(
            "This joins Bitcoin ETF flow context to current BTC perp state. It is not a trade instruction.\n\n"
        )
        handle.write("| asset | latest date | ETF action | latest BTC | 5d BTC | funding | OI notional | action | score |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |\n")
        for context in contexts:
            handle.write(
                f"| {context.asset} | {context.latest_date} | {context.etf_action} | "
                f"{context.latest_flow_btc:.8f} | {context.rolling_5d_flow_btc:.8f} | "
                f"{context.annualized_funding:.8f} | {context.open_interest_notional:.8f} | "
                f"{context.action} | {context.score:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "ETF demand can support a slower BTC regime, while perp funding shows faster leveraged positioning. The useful test is whether their agreement or conflict improves forward BTC labels.\n"
        )
    return output_path


def _action_for_context(*, etf_action: str, funding: float) -> str:
    if ("accumulation" in etf_action or "inflow" in etf_action) and funding > 0.0:
        return "etf_inflow_leverage_aligned_watch"
    if ("accumulation" in etf_action or "inflow" in etf_action) and funding < 0.0:
        return "etf_inflow_vs_short_perp_divergence_watch"
    if ("distribution" in etf_action or "outflow" in etf_action) and funding > 0.0:
        return "etf_outflow_vs_long_perp_divergence_watch"
    if ("distribution" in etf_action or "outflow" in etf_action) and funding < 0.0:
        return "etf_outflow_leverage_aligned_watch"
    return "etf_flow_market_context"


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
        "--snapshot-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_btc_etf_flow_snapshot.csv",
    )
    parser.add_argument(
        "--hyperliquid-path",
        type=Path,
        default=ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_btc_etf_market_join.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_btc_etf_market_join.md",
    )
    args = parser.parse_args()

    contexts = join_btc_etf_to_market(
        snapshot_path=args.snapshot_path,
        hyperliquid_path=args.hyperliquid_path,
    )
    write_contexts(contexts, output_path=args.output_path)
    write_markdown(contexts, output_path=args.markdown_output_path)
    for context in contexts:
        print(
            context.asset,
            context.action,
            f"latest_btc={context.latest_flow_btc:.2f}",
            f"five_day_btc={context.rolling_5d_flow_btc:.2f}",
            f"funding={context.annualized_funding:.6f}",
            f"score={context.score:.4f}",
        )


if __name__ == "__main__":
    main()
