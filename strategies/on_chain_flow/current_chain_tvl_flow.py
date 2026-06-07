from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import requests


CHAINS_URL = "https://api.llama.fi/v2/chains"
HISTORICAL_CHAIN_TVL_URL = "https://api.llama.fi/v2/historicalChainTvl"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ChainTvlFlowRow:
    timestamp: str
    chain: str
    token_symbol: str
    current_tvl_usd: float
    day_change_usd: float
    week_change_usd: float
    month_change_usd: float
    day_change_pct: float
    week_change_pct: float
    month_change_pct: float
    action: str
    followup: str


def build_chain_tvl_flow_rows(
    *,
    top_chains: int = 40,
    min_tvl_usd: float = 50_000_000.0,
) -> tuple[ChainTvlFlowRow, ...]:
    observed_at = datetime.now(UTC).isoformat()
    chain_rows = tuple(
        row
        for row in _fetch_chains()
        if float(row.get("tvl") or "0") >= min_tvl_usd
    )
    chain_rows = tuple(sorted(chain_rows, key=lambda row: float(row.get("tvl") or "0"), reverse=True))
    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = tuple(
            row
            for row in executor.map(
                lambda item: _build_row(item, timestamp=observed_at),
                chain_rows[:top_chains],
            )
            if row is not None
        )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_chain_tvl_flow_csv(
    rows: tuple[ChainTvlFlowRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "chain",
                "token_symbol",
                "current_tvl_usd",
                "day_change_usd",
                "week_change_usd",
                "month_change_usd",
                "day_change_pct",
                "week_change_pct",
                "month_change_pct",
                "action",
                "followup",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.chain,
                    row.token_symbol,
                    f"{row.current_tvl_usd:.2f}",
                    f"{row.day_change_usd:.2f}",
                    f"{row.week_change_usd:.2f}",
                    f"{row.month_change_usd:.2f}",
                    f"{row.day_change_pct:.8f}",
                    f"{row.week_change_pct:.8f}",
                    f"{row.month_change_pct:.8f}",
                    row.action,
                    row.followup,
                )
            )
    return output_path


def write_chain_tvl_flow_md(
    rows: tuple[ChainTvlFlowRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Chain TVL Flow\n\n")
        handle.write(
            "This ranks current chain-level TVL flow. It is a broad capital-flow "
            "screen, not a causal alpha test.\n\n"
        )
        handle.write(
            "| chain | token | tvl USD | day % | week % | month % | action | followup |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.chain} | "
                f"{row.token_symbol} | "
                f"{row.current_tvl_usd:.0f} | "
                f"{row.day_change_pct:.4f} | "
                f"{row.week_change_pct:.4f} | "
                f"{row.month_change_pct:.4f} | "
                f"{row.action} | "
                f"{row.followup} |\n"
            )
    return output_path


def _build_row(raw: dict[str, object], *, timestamp: str) -> ChainTvlFlowRow | None:
    chain = str(raw.get("name") or "")
    history = _fetch_historical_tvl(chain)
    if len(history) < 31:
        return None
    current = float(history[-1]["tvl"])
    prev_day = float(history[-2]["tvl"])
    prev_week = float(history[-8]["tvl"])
    prev_month = float(history[-31]["tvl"])
    day_change_usd = current - prev_day
    week_change_usd = current - prev_week
    month_change_usd = current - prev_month
    day_change_pct = _pct_change(current, prev_day)
    week_change_pct = _pct_change(current, prev_week)
    month_change_pct = _pct_change(current, prev_month)
    action = _action(day_change_pct=day_change_pct, week_change_pct=week_change_pct)
    return ChainTvlFlowRow(
        timestamp=timestamp,
        chain=chain,
        token_symbol=str(raw.get("tokenSymbol") or ""),
        current_tvl_usd=current,
        day_change_usd=day_change_usd,
        week_change_usd=week_change_usd,
        month_change_usd=month_change_usd,
        day_change_pct=day_change_pct,
        week_change_pct=week_change_pct,
        month_change_pct=month_change_pct,
        action=action,
        followup=_followup(action=action, token_symbol=str(raw.get("tokenSymbol") or "")),
    )


def _fetch_chains() -> tuple[dict[str, object], ...]:
    response = requests.get(CHAINS_URL, timeout=30)
    response.raise_for_status()
    return tuple(response.json())


def _fetch_historical_tvl(chain: str) -> tuple[dict[str, object], ...]:
    response = requests.get(
        f"{HISTORICAL_CHAIN_TVL_URL}/{quote(chain, safe='')}",
        timeout=30,
    )
    if response.status_code == 404:
        return ()
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        return ()
    return tuple(sorted(rows, key=lambda row: int(row["date"])))


def _pct_change(current: float, previous: float) -> float:
    if previous <= 0.0:
        return 0.0
    return (current / previous) - 1.0


def _action(*, day_change_pct: float, week_change_pct: float) -> str:
    if week_change_pct >= 0.05 and day_change_pct > 0.0:
        return "chain_inflow_momentum_watch"
    if week_change_pct <= -0.05 and day_change_pct < 0.0:
        return "chain_outflow_stress_watch"
    if abs(week_change_pct) >= 0.05:
        return "chain_flow_reversal_watch"
    return "chain_flow_context"


def _followup(*, action: str, token_symbol: str) -> str:
    if not token_symbol:
        return "use as broad on-chain regime context"
    if action == "chain_inflow_momentum_watch":
        return f"label {token_symbol} continuation against perp funding and liquidity"
    if action == "chain_outflow_stress_watch":
        return f"label {token_symbol} downside or rotation-away behavior"
    if action == "chain_flow_reversal_watch":
        return f"separate {token_symbol} reversal from stale TVL accounting"
    return "keep as context until flow strengthens"


def _sort_key(row: ChainTvlFlowRow) -> tuple[int, float, float]:
    priority = {
        "chain_inflow_momentum_watch": 3,
        "chain_outflow_stress_watch": 2,
        "chain_flow_reversal_watch": 1,
        "chain_flow_context": 0,
    }
    return (priority.get(row.action, 0), abs(row.week_change_pct), row.current_tvl_usd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-chains", type=int, default=40)
    parser.add_argument("--min-tvl-usd", type=float, default=50_000_000.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_chain_tvl_flow.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_chain_tvl_flow_rows(
        top_chains=args.top_chains,
        min_tvl_usd=args.min_tvl_usd,
    )
    write_chain_tvl_flow_csv(rows, output_path=args.output_path)
    write_chain_tvl_flow_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.chain,
            row.token_symbol,
            row.action,
            f"week={row.week_change_pct:.4f}",
            f"day={row.day_change_pct:.4f}",
        )


if __name__ == "__main__":
    main()
