from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import requests


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ClobDepthRow:
    market_id: str
    question: str
    outcome: str
    token_id: str
    best_bid: float
    best_ask: float
    spread: float
    top_bid_size: float
    top_ask_size: float
    bid_depth_to_5c: float
    ask_depth_to_5c: float
    visible_depth_score: float
    reason: str


def build_clob_depth_rows(
    *,
    monitor_summary_path: Path = ROOT / "current_polymarket_microstructure_monitor_summary.csv",
    top_markets: int = 20,
) -> tuple[ClobDepthRow, ...]:
    market_ids = _top_market_ids(monitor_summary_path, top=top_markets)
    rows: list[ClobDepthRow] = []
    for market_id in market_ids:
        market = _fetch_market(market_id)
        yes_token_id, no_token_id = _token_ids(market.get("clobTokenIds"))
        question = str(market.get("question") or "")
        rows.append(_try_build_depth_row(market_id=market_id, question=question, outcome="Yes", token_id=yes_token_id))
        rows.append(_try_build_depth_row(market_id=market_id, question=question, outcome="No", token_id=no_token_id))
    return tuple(sorted(rows, key=_depth_sort_key, reverse=True))


def write_clob_depth_csv(rows: tuple[ClobDepthRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "market_id",
                "question",
                "outcome",
                "token_id",
                "best_bid",
                "best_ask",
                "spread",
                "top_bid_size",
                "top_ask_size",
                "bid_depth_to_5c",
                "ask_depth_to_5c",
                "visible_depth_score",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.market_id,
                    row.question,
                    row.outcome,
                    row.token_id,
                    f"{row.best_bid:.6f}",
                    f"{row.best_ask:.6f}",
                    f"{row.spread:.6f}",
                    f"{row.top_bid_size:.6f}",
                    f"{row.top_ask_size:.6f}",
                    f"{row.bid_depth_to_5c:.6f}",
                    f"{row.ask_depth_to_5c:.6f}",
                    f"{row.visible_depth_score:.8f}",
                    row.reason,
                )
            )
    return output_path


def write_clob_depth_md(
    rows: tuple[ClobDepthRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Polymarket CLOB Depth\n\n")
        handle.write(
            "This checks visible CLOB depth for unsettled current microstructure monitor "
            "markets first, then falls back to near-certain markets only if needed. "
            "It is not a trade instruction.\n\n"
        )
        handle.write(
            "| question | outcome | bid | ask | spread | top bid size | top ask size | bid depth 5c | ask depth 5c | score | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{_escape(row.question)} | "
                f"{row.outcome} | "
                f"{row.best_bid:.4f} | "
                f"{row.best_ask:.4f} | "
                f"{row.spread:.4f} | "
                f"{row.top_bid_size:.2f} | "
                f"{row.top_ask_size:.2f} | "
                f"{row.bid_depth_to_5c:.2f} | "
                f"{row.ask_depth_to_5c:.2f} | "
                f"{row.visible_depth_score:.4f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Depth is measured in outcome-token size, not guaranteed executable USD. "
            "A high score means visible public depth exists near top of book; it does "
            "not prove queue priority, fill probability, or adverse-selection edge.\n"
        )
    return output_path


def _top_market_ids(path: Path, *, top: int) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    unsettled = tuple(row for row in rows if 0.05 < _float(row.get("mean_midpoint")) < 0.95)
    near_certain = tuple(row for row in rows if row not in unsettled)
    selected = (*unsettled[:top], *near_certain[: max(top - len(unsettled), 0)])
    return tuple(row["market_id"] for row in selected[:top])


def _fetch_market(market_id: str) -> dict[str, object]:
    response = requests.get(f"{GAMMA_BASE_URL}/markets/{market_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def _fetch_book(token_id: str) -> dict[str, object]:
    response = requests.get(f"{CLOB_BASE_URL}/book", params={"token_id": token_id}, timeout=30)
    response.raise_for_status()
    return response.json()


def _build_depth_row(
    *,
    market_id: str,
    question: str,
    outcome: str,
    token_id: str,
) -> ClobDepthRow:
    book = _fetch_book(token_id)
    bids = _levels(book.get("bids"))
    asks = _levels(book.get("asks"))
    best_bid = bids[-1][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    spread = best_ask - best_bid if best_ask > 0.0 and best_bid > 0.0 else 0.0
    top_bid_size = bids[-1][1] if bids else 0.0
    top_ask_size = asks[0][1] if asks else 0.0
    bid_depth_to_5c = _depth_near_top(bids, best_bid, side="bid")
    ask_depth_to_5c = _depth_near_top(asks, best_ask, side="ask")
    return ClobDepthRow(
        market_id=market_id,
        question=question,
        outcome=outcome,
        token_id=token_id,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        top_bid_size=top_bid_size,
        top_ask_size=top_ask_size,
        bid_depth_to_5c=bid_depth_to_5c,
        ask_depth_to_5c=ask_depth_to_5c,
        visible_depth_score=(min(bid_depth_to_5c, ask_depth_to_5c) / 1_000.0) - (spread * 10.0),
        reason=_reason(bid_depth_to_5c=bid_depth_to_5c, ask_depth_to_5c=ask_depth_to_5c, spread=spread),
    )


def _try_build_depth_row(
    *,
    market_id: str,
    question: str,
    outcome: str,
    token_id: str,
) -> ClobDepthRow:
    if not token_id:
        return _unavailable_depth_row(
            market_id=market_id,
            question=question,
            outcome=outcome,
            token_id=token_id,
            reason="missing CLOB token id",
        )
    try:
        return _build_depth_row(market_id=market_id, question=question, outcome=outcome, token_id=token_id)
    except requests.RequestException as exc:
        status_code = getattr(exc.response, "status_code", None)
        suffix = f"HTTP {status_code}" if status_code is not None else exc.__class__.__name__
        return _unavailable_depth_row(
            market_id=market_id,
            question=question,
            outcome=outcome,
            token_id=token_id,
            reason=f"CLOB book unavailable: {suffix}",
        )


def _unavailable_depth_row(
    *,
    market_id: str,
    question: str,
    outcome: str,
    token_id: str,
    reason: str,
) -> ClobDepthRow:
    return ClobDepthRow(
        market_id=market_id,
        question=question,
        outcome=outcome,
        token_id=token_id,
        best_bid=0.0,
        best_ask=0.0,
        spread=0.0,
        top_bid_size=0.0,
        top_ask_size=0.0,
        bid_depth_to_5c=0.0,
        ask_depth_to_5c=0.0,
        visible_depth_score=0.0,
        reason=reason,
    )


def _levels(value: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        sorted(
            (float(level["price"]), float(level["size"]))
            for level in value
            if float(level.get("price") or 0.0) > 0.0
        )
    )


def _depth_near_top(levels: tuple[tuple[float, float], ...], top_price: float, *, side: str) -> float:
    if top_price <= 0.0:
        return 0.0
    if side == "bid":
        return sum(size for price, size in levels if price >= top_price - 0.05)
    return sum(size for price, size in levels if price <= top_price + 0.05)


def _reason(*, bid_depth_to_5c: float, ask_depth_to_5c: float, spread: float) -> str:
    if bid_depth_to_5c >= 1_000.0 and ask_depth_to_5c >= 1_000.0 and spread <= 0.05:
        return "visible depth exists near both sides"
    if bid_depth_to_5c < 1_000.0 or ask_depth_to_5c < 1_000.0:
        return "visible near-top depth is thin"
    return "spread is wide despite visible depth"


def _depth_sort_key(row: ClobDepthRow) -> tuple[int, float]:
    return (0 if _is_near_certain_book(row) else 1, row.visible_depth_score)


def _is_near_certain_book(row: ClobDepthRow) -> bool:
    if row.best_bid <= 0.0 or row.best_ask <= 0.0:
        return True
    midpoint = (row.best_bid + row.best_ask) / 2.0
    return midpoint <= 0.05 or midpoint >= 0.95


def _token_ids(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, list):
        parsed = value
    else:
        parsed = []
    yes_token_id = str(parsed[0]) if len(parsed) > 0 else ""
    no_token_id = str(parsed[1]) if len(parsed) > 1 else ""
    return yes_token_id, no_token_id


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monitor-summary-path",
        type=Path,
        default=ROOT / "current_polymarket_microstructure_monitor_summary.csv",
    )
    parser.add_argument("--top-markets", type=int, default=20)
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "current_polymarket_clob_depth.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_polymarket_clob_depth.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_clob_depth_rows(
        monitor_summary_path=args.monitor_summary_path,
        top_markets=args.top_markets,
    )
    write_clob_depth_csv(rows, output_path=args.csv_output_path)
    write_clob_depth_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.outcome,
            f"spread={row.spread:.4f}",
            f"bid_depth_5c={row.bid_depth_to_5c:.0f}",
            f"ask_depth_5c={row.ask_depth_to_5c:.0f}",
            row.question,
        )


if __name__ == "__main__":
    main()
