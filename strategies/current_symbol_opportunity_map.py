from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"[A-Za-z$][A-Za-z0-9$]{1,18}")
STOP_TOKENS = {
    "ACTION",
    "ACTIONABILITY",
    "ACTIVITY",
    "ANNOUNCE",
    "ANNOUNCEMENT",
    "AGE",
    "ALLOCATE",
    "AND",
    "ANOMALY",
    "APY",
    "APYX",
    "ASK",
    "ASSETS",
    "ATM",
    "ATTENTION",
    "AFTER",
    "AVOID",
    "BASE",
    "BASIS",
    "BEFORE",
    "BETA",
    "BITCOIN",
    "BID",
    "BIN",
    "BINANCE",
    "BINPERP",
    "BLUE",
    "BORROW",
    "BPS",
    "BYBIT",
    "BYBITPERP",
    "CALENDAR",
    "BRIDGE",
    "CANDIDATE",
    "CAPITAL",
    "CARRY",
    "CASH",
    "CASCADE",
    "CEX",
    "CHAIN",
    "CHECK",
    "CIRCLE",
    "COINDESK",
    "COLLECT",
    "COINUP",
    "CONTEXT",
    "CONTINUATION",
    "CROWDED",
    "CROWDING",
    "CROSS",
    "CRYPTO",
    "CRYPTO_EVENT",
    "CURRENT",
    "CURVE",
    "COVER",
    "DEPTH",
    "DECRYPT",
    "DEX",
    "DIRECTIONAL",
    "DISLOCATION",
    "DOLLAR",
    "EDGE",
    "ENTRY",
    "EVENT",
    "EXEC",
    "EXECUTABLE",
    "EXECUTION",
    "EXCHANGE",
    "EXIT",
    "EXPOSURE",
    "EXTREME",
    "FAILED",
    "FEE",
    "FEES",
    "FLUID",
    "FOLLOW",
    "FLOW",
    "FUNDING",
    "FUTURE",
    "FUTURES",
    "GATE",
    "GEOPOLITICAL",
    "GEOPOLITICAL_EVENT",
    "GROWTH",
    "HEDGE",
    "HL",
    "HLPERP",
    "IMBALANCE",
    "IMPACT",
    "INDEX",
    "INSTITUTIONAL",
    "IO",
    "JUPITER",
    "KUCOIN",
    "LAG",
    "LABEL",
    "LBANK",
    "LEND",
    "LENDING",
    "L2",
    "LIQUIDATION",
    "LONG",
    "LOSS",
    "MACRO",
    "MARK",
    "MARKET",
    "MECHANICS",
    "MIGRATION",
    "MODEL",
    "MOMENTUM",
    "NEWS",
    "NEW",
    "NO",
    "NONE",
    "OKX",
    "OKXSWAP",
    "ON",
    "ONLY",
    "OFF",
    "OI",
    "OR",
    "ORACLE",
    "OUTCOME",
    "PAPER",
    "OVERLAP",
    "PEG",
    "PENDING",
    "PERP",
    "PERPETUAL",
    "PATH",
    "POOL",
    "PNL",
    "PREMIUM",
    "PREDICTION",
    "PROBABILITY",
    "PRESSURE",
    "PRICE",
    "POSITIONING",
    "POLITICAL",
    "POLITICAL_EVENT",
    "PROBE",
    "PROXY",
    "PROTOCOL",
    "PUMPSWAP",
    "PUT",
    "REPEAT",
    "RE",
    "RELATED",
    "RESEARCH",
    "RELATIVE",
    "REPEG",
    "REVERSION",
    "REVERSAL",
    "RISK",
    "SCORE",
    "SECURITY",
    "SHORT",
    "SPOT",
    "SQUEEZE",
    "SOLANA",
    "SOURCE",
    "SPREAD",
    "SPORTS",
    "SPORTS_EVENT",
    "STABLECOIN",
    "STACK",
    "STATUS",
    "STRESS",
    "STRADDLE",
    "STRONGLY",
    "SUPPORTED",
    "SWAP",
    "SUPPLY",
    "TERM",
    "TOKEN",
    "TRADE",
    "TRADEABILITY",
    "TVL",
    "UNLOCK",
    "UNTIL",
    "UNWIND",
    "UNISWAP",
    "USD",
    "US",
    "USDC",
    "USDT",
    "VALUE",
    "VALIDATED",
    "VALUATION",
    "VENUE",
    "VOL",
    "VOLATILITY",
    "VOLUME",
    "V3",
    "VS",
    "WATCH",
    "WHITEBIT",
    "WILL",
    "WIN",
    "XYZ",
    "YIELD",
    "YES",
    "ZCASH",
}
CHAIN_NAMES = {
    "ARBITRUM",
    "AVALANCHE",
    "BASE",
    "BERACHAIN",
    "BSC",
    "ETHEREUM",
    "HYPERLIQUID",
    "MANTLE",
    "OPTIMISM",
    "PLASMA",
    "POLYGON",
    "SEI",
    "SOLANA",
    "STELLAR",
    "SUI",
    "TRON",
}


@dataclass(frozen=True)
class SymbolOpportunityRow:
    symbol: str
    status: str
    cluster_score: float
    source_count: int
    candidate_count: int
    max_priority: float
    mean_priority: float
    sources: str
    top_opportunities: str
    sides: str
    main_conflict: str
    next_step: str


def build_symbol_opportunity_rows(
    *,
    stack_path: Path = ROOT / "current_alpha_stack.csv",
) -> tuple[SymbolOpportunityRow, ...]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_rows(stack_path):
        for symbol in _symbols_for_stack_row(row):
            grouped.setdefault(symbol, []).append(row)
    rows = tuple(_build_symbol_row(symbol=symbol, rows=rows) for symbol, rows in grouped.items())
    return tuple(sorted(rows, key=lambda row: row.cluster_score, reverse=True))


def write_symbol_opportunity_csv(
    rows: tuple[SymbolOpportunityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "status",
                "cluster_score",
                "source_count",
                "candidate_count",
                "max_priority",
                "mean_priority",
                "sources",
                "top_opportunities",
                "sides",
                "main_conflict",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.status,
                    f"{row.cluster_score:.8f}",
                    row.source_count,
                    row.candidate_count,
                    f"{row.max_priority:.8f}",
                    f"{row.mean_priority:.8f}",
                    row.sources,
                    row.top_opportunities,
                    row.sides,
                    row.main_conflict,
                    row.next_step,
                )
            )
    return output_path


def write_symbol_opportunity_md(
    rows: tuple[SymbolOpportunityRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Symbol Opportunity Map\n\n")
        handle.write(
            "This groups the current alpha stack by tradable or research symbol. "
            "It is a prioritization map for repeated labels and execution checks, not a trade list.\n\n"
        )
        handle.write(
            "| symbol | status | score | sources | candidates | max priority | top opportunities | sides | next step |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.status} | {row.cluster_score:.4f} | "
                f"{row.source_count} | {row.candidate_count} | {row.max_priority:.4f} | "
                f"{_escape(row.top_opportunities)} | {_escape(row.sides)} | {_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Higher rows have either high-priority single candidates or multiple independent lanes pointing at the same symbol. "
            "The next useful step is to label these clusters against forward returns, funding PnL, depth, fees, and execution feasibility.\n"
        )
    return output_path


def _build_symbol_row(*, symbol: str, rows: list[dict[str, str]]) -> SymbolOpportunityRow:
    sorted_rows = sorted(rows, key=lambda row: _float(row.get("priority_score")), reverse=True)
    priorities = tuple(_float(row.get("priority_score")) for row in sorted_rows)
    sources = tuple(sorted({source.strip() for row in sorted_rows for source in row.get("sources", "").split("+") if source.strip()}))
    opportunities = tuple(row.get("opportunity", "") for row in sorted_rows[:5])
    sides = tuple(_unique(row.get("side", "") for row in sorted_rows if row.get("side", "")))[:5]
    status = _cluster_status(source_count=len(sources), candidate_count=len(sorted_rows))
    max_priority = max(priorities) if priorities else 0.0
    mean_priority = sum(priorities) / len(priorities) if priorities else 0.0
    cluster_score = max_priority + min(len(sources) * 4.0, 16.0) + min(len(sorted_rows) * 1.5, 12.0)
    return SymbolOpportunityRow(
        symbol=symbol,
        status=status,
        cluster_score=cluster_score,
        source_count=len(sources),
        candidate_count=len(sorted_rows),
        max_priority=max_priority,
        mean_priority=mean_priority,
        sources=" + ".join(sources),
        top_opportunities=", ".join(opportunities),
        sides=", ".join(sides),
        main_conflict=sorted_rows[0].get("conflict", "") if sorted_rows else "",
        next_step=_cluster_next_step(symbol=symbol, status=status, rows=sorted_rows),
    )


def _cluster_status(*, source_count: int, candidate_count: int) -> str:
    if source_count >= 3:
        return "cross_lane_cluster"
    if source_count == 2:
        return "two_lane_cluster"
    if candidate_count >= 3:
        return "single_lane_repeat_cluster"
    return "single_candidate"


def _cluster_next_step(*, symbol: str, status: str, rows: list[dict[str, str]]) -> str:
    if status in {"cross_lane_cluster", "two_lane_cluster"}:
        return f"build a {symbol} cluster label: forward return, funding PnL, depth, fees, and conflict between lanes"
    return rows[0].get("next_step", f"label {symbol} forward return and execution feasibility") if rows else ""


def _symbols_for_stack_row(row: dict[str, str]) -> tuple[str, ...]:
    symbols: set[str] = set()
    symbols.update(_symbols_from_evidence_head(row.get("evidence", "")))
    symbols.update(_symbols_from_free_text(row.get("opportunity", "")))
    side = row.get("side", "")
    if " " not in side and "?" not in side:
        symbols.update(_symbols_from_free_text(side))
    return tuple(sorted(symbols))


def _symbols_from_evidence_head(evidence: str) -> tuple[str, ...]:
    head = re.split(r"[;,]", evidence.split(":", 1)[0])[0]
    raw_tokens: list[str] = []
    if "/" in head:
        for part in head.split("/"):
            raw_tokens.extend(TOKEN_RE.findall(part)[:3])
    else:
        raw_tokens.extend(TOKEN_RE.findall(head)[:4])
    symbols: set[str] = set()
    for token in raw_tokens:
        normalized = _normalize_symbol(token)
        if normalized:
            symbols.add(normalized)
    return tuple(sorted(symbols))


def _symbols_from_free_text(value: str) -> tuple[str, ...]:
    symbols: set[str] = set()
    for token in TOKEN_RE.findall(value.replace("-", "_")):
        normalized = _normalize_symbol(token)
        if normalized:
            symbols.add(normalized)
    return tuple(sorted(symbols))


def _normalize_symbol(token: str) -> str:
    value = token.strip().strip("_;:,").strip("$").upper()
    if not value:
        return ""
    value = value.replace(":", "_").replace("-", "_")
    if value in STOP_TOKENS:
        return ""
    for suffix in ("_PERP", "_SWAP", "PERP", "USDTM", "USDT"):
        if value.endswith(suffix) and len(value) > len(suffix) + 1:
            value = value[: -len(suffix)]
            break
    value = value.strip("_")
    if (
        not value
        or value in STOP_TOKENS
        or value in CHAIN_NAMES
        or re.fullmatch(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}", value)
        or value[0].isdigit()
        or value.isdigit()
        or len(value) > 12
    ):
        return ""
    if len(value) == 1 and value not in {"H"}:
        return ""
    return value


def _unique(values: object) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-path", type=Path, default=ROOT / "current_alpha_stack.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_symbol_opportunity_map.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_symbol_opportunity_map.md")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_symbol_opportunity_rows(stack_path=args.stack_path)
    write_symbol_opportunity_csv(rows, output_path=args.output_path)
    write_symbol_opportunity_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.status,
            row.symbol,
            f"sources={row.source_count}",
            f"candidates={row.candidate_count}",
            f"score={row.cluster_score:.4f}",
        )


if __name__ == "__main__":
    main()
