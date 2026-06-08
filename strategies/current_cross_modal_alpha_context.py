from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModalEvidence:
    symbol: str
    source: str
    direction: int
    score: float
    evidence: str
    missing_work: str
    next_step: str


@dataclass(frozen=True)
class CrossModalAlphaContextRow:
    symbol: str
    decision: str
    aligned_direction: str
    total_score: float
    source_count: int
    aligned_sources: str
    conflicting_sources: str
    evidence: str
    missing_work: str
    next_step: str


def build_cross_modal_alpha_context(
    *,
    event_pressure_path: Path = ROOT / "news_social" / "current_event_pressure_cluster.csv",
    stablecoin_migration_path: Path = ROOT / "stablecoin_liquidity" / "current_chain_stablecoin_migration.csv",
    wallet_actionability_path: Path = ROOT / "wallet_entity_flow" / "current_seed_wallet_flow_actionability.csv",
    chain_context_path: Path = ROOT / "on_chain_flow" / "current_chain_tvl_flow_market_context.csv",
    dex_pool_flow_path: Path = ROOT / "dex_pool_flow" / "current_geckoterminal_pool_flow.csv",
) -> tuple[CrossModalAlphaContextRow, ...]:
    evidence = collect_cross_modal_evidence(
        event_pressure_path=event_pressure_path,
        stablecoin_migration_path=stablecoin_migration_path,
        wallet_actionability_path=wallet_actionability_path,
        chain_context_path=chain_context_path,
        dex_pool_flow_path=dex_pool_flow_path,
    )
    grouped: dict[str, list[ModalEvidence]] = {}
    for row in evidence:
        if row.symbol:
            grouped.setdefault(row.symbol, []).append(row)
    rows = tuple(_context_row(symbol=symbol, evidence=tuple(rows)) for symbol, rows in grouped.items())
    useful_rows = tuple(row for row in rows if row.source_count >= 2)
    return tuple(sorted(useful_rows, key=lambda row: row.total_score, reverse=True))


def collect_cross_modal_evidence(
    *,
    event_pressure_path: Path = ROOT / "news_social" / "current_event_pressure_cluster.csv",
    stablecoin_migration_path: Path = ROOT / "stablecoin_liquidity" / "current_chain_stablecoin_migration.csv",
    wallet_actionability_path: Path = ROOT / "wallet_entity_flow" / "current_seed_wallet_flow_actionability.csv",
    chain_context_path: Path = ROOT / "on_chain_flow" / "current_chain_tvl_flow_market_context.csv",
    dex_pool_flow_path: Path = ROOT / "dex_pool_flow" / "current_geckoterminal_pool_flow.csv",
) -> tuple[ModalEvidence, ...]:
    return (
        _event_evidence(event_pressure_path)
        + _stablecoin_evidence(stablecoin_migration_path)
        + _wallet_evidence(wallet_actionability_path)
        + _chain_context_evidence(chain_context_path)
        + _dex_pool_evidence(dex_pool_flow_path)
    )


def write_cross_modal_alpha_context_csv(
    rows: tuple[CrossModalAlphaContextRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "decision",
                "aligned_direction",
                "total_score",
                "source_count",
                "aligned_sources",
                "conflicting_sources",
                "evidence",
                "missing_work",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.decision,
                    row.aligned_direction,
                    f"{row.total_score:.8f}",
                    row.source_count,
                    row.aligned_sources,
                    row.conflicting_sources,
                    row.evidence,
                    row.missing_work,
                    row.next_step,
                )
            )
    return output_path


def write_cross_modal_alpha_context_md(
    rows: tuple[CrossModalAlphaContextRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Cross-Modal Alpha Context\n\n")
        handle.write(
            "This joins event, stablecoin, wallet, chain-flow, and DEX-pool context by tradable asset. "
            "It is a context join for alpha discovery, not a trade list or a strategy abstraction.\n\n"
        )
        handle.write(
            "| symbol | decision | direction | score | sources | conflicts | evidence | missing work | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | "
                f"{row.decision} | "
                f"{row.aligned_direction} | "
                f"{row.total_score:.4f} | "
                f"{row.source_count} | "
                f"{_escape(row.conflicting_sources)} | "
                f"{_escape(row.evidence)} | "
                f"{_escape(row.missing_work)} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _event_evidence(path: Path) -> tuple[ModalEvidence, ...]:
    rows = []
    for row in _read_rows(path):
        symbol = _symbol(row.get("symbol", ""))
        direction = _direction(row.get("side", ""))
        if direction == 0:
            continue
        rows.append(
            ModalEvidence(
                symbol=symbol,
                source="event_pressure",
                direction=direction,
                score=_float(row.get("score")),
                evidence=(
                    f"event={row.get('status', '')}; sources={row.get('source_count', '')}; "
                    f"events={row.get('event_count', '')}; top={row.get('top_sources', '')}"
                ),
                missing_work="event timestamp quality, duplicate-source filtering, beta attribution, and execution check",
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _stablecoin_evidence(path: Path) -> tuple[ModalEvidence, ...]:
    rows = []
    for row in _read_rows(path):
        symbol = _symbol(row.get("token_symbol", ""))
        direction = _direction(row.get("side", ""))
        if not symbol or direction == 0:
            continue
        rows.append(
            ModalEvidence(
                symbol=symbol,
                source="stablecoin_migration",
                direction=direction,
                score=_float(row.get("score")),
                evidence=(
                    f"stablecoin={row.get('status', '')}; chain={row.get('chain', '')}; "
                    f"week_pct={row.get('week_change_pct', '')}; top_asset={row.get('top_asset', '')}"
                ),
                missing_work="bridge route, chain-token mapping, venue coverage, and beta attribution",
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _wallet_evidence(path: Path) -> tuple[ModalEvidence, ...]:
    rows = []
    for row in _read_rows(path):
        status = row.get("status", "")
        if "reject" in status or "blocked" in status:
            continue
        symbol = _symbol(row.get("execution_asset", ""))
        direction = _direction(row.get("side", ""))
        if not symbol or direction == 0:
            continue
        rows.append(
            ModalEvidence(
                symbol=symbol,
                source="wallet_flow",
                direction=direction,
                score=_float(row.get("score")),
                evidence=(
                    f"wallet={row.get('wallet_label', '')}; status={status}; "
                    f"position_notional={row.get('current_position_notional', '')}; pnl={row.get('net_closed_pnl_after_fees', '')}"
                ),
                missing_work="wallet source quality, copycat risk, forward labels, funding, spread/depth, and survivorship bias",
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _chain_context_evidence(path: Path) -> tuple[ModalEvidence, ...]:
    rows = []
    for row in _read_rows(path):
        symbol = _symbol(row.get("token_symbol", ""))
        direction = int(_float(row.get("direction")))
        score = max(_float(row.get("context_score")) * 100.0, 0.0)
        if not symbol or direction == 0 or score <= 0.0:
            continue
        rows.append(
            ModalEvidence(
                symbol=symbol,
                source=f"chain_context_{row.get('venue', '').lower()}",
                direction=direction,
                score=score,
                evidence=(
                    f"chain_context={row.get('action', '')}; venue={row.get('venue', '')}; "
                    f"funding_support={row.get('funding_support', '')}; context={row.get('context_score', '')}"
                ),
                missing_work="repeat labels, venue-specific funding PnL, depth, fees, and liquidation context",
                next_step="label chain-flow context by venue and compare funding/depth support before paper probing",
            )
        )
    return tuple(rows)


def _dex_pool_evidence(path: Path) -> tuple[ModalEvidence, ...]:
    rows = []
    for row in _read_rows(path):
        symbol = _pool_symbol(row.get("name", ""))
        direction = _direction(row.get("side", ""))
        if not symbol or direction == 0:
            continue
        rows.append(
            ModalEvidence(
                symbol=symbol,
                source="dex_pool_flow",
                direction=direction,
                score=_float(row.get("score")),
                evidence=(
                    f"dex={row.get('network', '')}/{row.get('dex', '')}; status={row.get('status', '')}; "
                    f"vol_reserve={row.get('volume_reserve_ratio_h1', '')}; change_h1={row.get('price_change_h1', '')}"
                ),
                missing_work="route simulation, slippage, gas, MEV, token restrictions, and repeat-flow labels",
                next_step=row.get("next_step", ""),
            )
        )
    return tuple(rows)


def _context_row(*, symbol: str, evidence: tuple[ModalEvidence, ...]) -> CrossModalAlphaContextRow:
    long_rows = tuple(row for row in evidence if row.direction > 0)
    short_rows = tuple(row for row in evidence if row.direction < 0)
    aligned_rows = long_rows if _score(long_rows) >= _score(short_rows) else short_rows
    conflicting_rows = short_rows if aligned_rows is long_rows else long_rows
    aligned_direction = "long" if aligned_rows and aligned_rows[0].direction > 0 else "short"
    total_score = _score(aligned_rows) + min(len(aligned_rows) * 12.0, 48.0) - min(_score(conflicting_rows) * 0.35, 80.0)
    decision = _decision(aligned_rows=aligned_rows, conflicting_rows=conflicting_rows, total_score=total_score)
    return CrossModalAlphaContextRow(
        symbol=symbol,
        decision=decision,
        aligned_direction=aligned_direction,
        total_score=total_score,
        source_count=len(evidence),
        aligned_sources=", ".join(row.source for row in aligned_rows),
        conflicting_sources=", ".join(row.source for row in conflicting_rows),
        evidence=" || ".join(row.evidence for row in aligned_rows[:5]),
        missing_work="; ".join(dict.fromkeys(row.missing_work for row in aligned_rows[:5])),
        next_step=_next_step(symbol=symbol, decision=decision, direction=aligned_direction, aligned_rows=aligned_rows),
    )


def _decision(
    *,
    aligned_rows: tuple[ModalEvidence, ...],
    conflicting_rows: tuple[ModalEvidence, ...],
    total_score: float,
) -> str:
    if conflicting_rows and len(aligned_rows) < 3:
        return "split_conflicting_modal_context"
    if len(aligned_rows) >= 3 and total_score >= 170.0 and not conflicting_rows:
        return "cross_modal_probe_now"
    if len(aligned_rows) >= 3 and total_score >= 150.0:
        return "cross_modal_probe_after_conflict_check"
    if len(aligned_rows) >= 2 and total_score >= 100.0:
        return "label_cross_modal_context"
    return "watch_cross_modal_context"


def _next_step(
    *,
    symbol: str,
    decision: str,
    direction: str,
    aligned_rows: tuple[ModalEvidence, ...],
) -> str:
    sources = ", ".join(row.source for row in aligned_rows)
    if decision == "cross_modal_probe_now":
        return f"open a small {symbol} {direction} cross-modal paper label with source timestamps, funding, depth, and beta controls"
    if decision == "cross_modal_probe_after_conflict_check":
        return f"split {symbol} by modal source, then paper-label the {direction} context if conflicts are explained"
    if decision == "label_cross_modal_context":
        return f"label {symbol} {direction} context over 15m/1h/4h using sources: {sources}"
    if decision == "split_conflicting_modal_context":
        return f"do not collapse {symbol}; label conflicting modal sources separately"
    return f"keep collecting {symbol} cross-modal evidence before paper probing"


def _score(rows: tuple[ModalEvidence, ...]) -> float:
    return sum(row.score for row in rows)


def _direction(value: str) -> int:
    text = value.lower()
    if "short" in text or "outflow" in text or "reduce" in text:
        return -1
    if "long" in text or "inflow" in text or "activity" in text:
        return 1
    return 0


def _pool_symbol(value: str) -> str:
    if not value:
        return ""
    token = value.split("/")[0].strip().split()[0]
    return _symbol(token)


def _symbol(value: str) -> str:
    return value.strip().upper().replace(" ", "")


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_cross_modal_alpha_context.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_cross_modal_alpha_context.md")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_cross_modal_alpha_context()
    write_cross_modal_alpha_context_csv(rows, output_path=args.output_path)
    write_cross_modal_alpha_context_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.decision, row.aligned_direction, f"{row.total_score:.4f}")


if __name__ == "__main__":
    main()
