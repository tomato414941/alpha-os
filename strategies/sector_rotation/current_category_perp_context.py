from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class CategoryPerpContextRow:
    timestamp: str
    category_name: str
    symbol: str
    category_action: str
    direction: int
    category_change_24h: float
    directional_return_15m: float | None
    hl_annualized_funding: float | None
    okx_annualized_funding: float | None
    best_funding_support: float | None
    hl_liquidity_usd: float | None
    okx_liquidity_usd: float | None
    context_score: float
    action: str
    reason: str


def build_category_perp_context_rows(
    *,
    label_path: Path = ROOT / "current_category_tradable_forward_labels.csv",
    hl_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
    okx_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_okx_perp_pressure.csv",
) -> tuple[CategoryPerpContextRow, ...]:
    hl_by_asset = {row["asset"]: row for row in _read_rows(hl_path)}
    okx_by_asset = {row["asset"]: row for row in _read_rows(okx_path)}
    rows = tuple(
        _build_row(
            label=row,
            hl_row=hl_by_asset.get(row["symbol"]),
            okx_row=okx_by_asset.get(row["symbol"]),
        )
        for row in _read_rows(label_path)
        if row.get("symbol")
        and (
            row["symbol"] in hl_by_asset
            or row["symbol"] in okx_by_asset
        )
    )
    return tuple(sorted(rows, key=lambda row: row.context_score, reverse=True))


def write_category_perp_context_csv(
    rows: tuple[CategoryPerpContextRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "category_name",
                "symbol",
                "category_action",
                "direction",
                "category_change_24h",
                "directional_return_15m",
                "hl_annualized_funding",
                "okx_annualized_funding",
                "best_funding_support",
                "hl_liquidity_usd",
                "okx_liquidity_usd",
                "context_score",
                "action",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.category_name,
                    row.symbol,
                    row.category_action,
                    row.direction,
                    f"{row.category_change_24h:.8f}",
                    _format_float(row.directional_return_15m),
                    _format_float(row.hl_annualized_funding),
                    _format_float(row.okx_annualized_funding),
                    _format_float(row.best_funding_support),
                    _format_float(row.hl_liquidity_usd),
                    _format_float(row.okx_liquidity_usd),
                    f"{row.context_score:.8f}",
                    row.action,
                    row.reason,
                )
            )
    return output_path


def write_category_perp_context_md(
    rows: tuple[CategoryPerpContextRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Category Perp Context\n\n")
        handle.write(
            "This joins category-rotation labels with current perp funding and "
            "liquidity. It is a research screen, not a deployable strategy.\n\n"
        )
        handle.write(
            "| category | symbol | dir | dir15 | funding support | HL funding | OKX funding | score | action | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.category_name} | "
                f"{row.symbol} | "
                f"{row.direction} | "
                f"{_format_float(row.directional_return_15m)} | "
                f"{_format_float(row.best_funding_support)} | "
                f"{_format_float(row.hl_annualized_funding)} | "
                f"{_format_float(row.okx_annualized_funding)} | "
                f"{row.context_score:.6f} | "
                f"{row.action} | "
                f"{row.reason} |\n"
            )
    return output_path


def _build_row(
    *,
    label: dict[str, str],
    hl_row: dict[str, str] | None,
    okx_row: dict[str, str] | None,
) -> CategoryPerpContextRow:
    direction = int(label.get("direction") or "0")
    hl_funding = _float_or_none((hl_row or {}).get("annualized_funding", ""))
    okx_funding = _float_or_none((okx_row or {}).get("annualized_funding", ""))
    funding_support_values = tuple(
        value for value in (
            _funding_support(direction=direction, annualized_funding=hl_funding),
            _funding_support(direction=direction, annualized_funding=okx_funding),
        )
        if value is not None
    )
    best_funding_support = max(funding_support_values) if funding_support_values else None
    directional_return_15m = _float_or_none(label.get("directional_return_15m", ""))
    hl_liquidity = _float_or_none((hl_row or {}).get("day_notional_volume", ""))
    okx_liquidity = _float_or_none((okx_row or {}).get("day_volume_usd", ""))
    context_score = _context_score(
        category_change_24h=float(label.get("category_change_24h") or "0"),
        directional_return_15m=directional_return_15m,
        best_funding_support=best_funding_support,
        hl_liquidity=hl_liquidity,
        okx_liquidity=okx_liquidity,
    )
    return CategoryPerpContextRow(
        timestamp=label["timestamp"],
        category_name=label["category_name"],
        symbol=label["symbol"],
        category_action=label["category_action"],
        direction=direction,
        category_change_24h=float(label.get("category_change_24h") or "0"),
        directional_return_15m=directional_return_15m,
        hl_annualized_funding=hl_funding,
        okx_annualized_funding=okx_funding,
        best_funding_support=best_funding_support,
        hl_liquidity_usd=hl_liquidity,
        okx_liquidity_usd=okx_liquidity,
        context_score=context_score,
        action=_action(
            directional_return_15m=directional_return_15m,
            best_funding_support=best_funding_support,
            context_score=context_score,
        ),
        reason=_reason(
            directional_return_15m=directional_return_15m,
            best_funding_support=best_funding_support,
        ),
    )


def _context_score(
    *,
    category_change_24h: float,
    directional_return_15m: float | None,
    best_funding_support: float | None,
    hl_liquidity: float | None,
    okx_liquidity: float | None,
) -> float:
    score = abs(category_change_24h) / 50.0
    if directional_return_15m is not None:
        score += directional_return_15m * 100.0
    if best_funding_support is not None:
        score += min(best_funding_support, 0.75)
        if best_funding_support <= 0.0:
            score -= 0.5
    if max(hl_liquidity or 0.0, okx_liquidity or 0.0) < 1_000_000.0:
        score -= 0.5
    return score


def _action(
    *,
    directional_return_15m: float | None,
    best_funding_support: float | None,
    context_score: float,
) -> str:
    if directional_return_15m is None:
        return "wait_for_label"
    if directional_return_15m > 0.0 and (best_funding_support or 0.0) > 0.0:
        return "sector_perp_repeat_candidate"
    if context_score > 1.0:
        return "keep_sampling"
    return "deprioritize"


def _reason(
    *,
    directional_return_15m: float | None,
    best_funding_support: float | None,
) -> str:
    if directional_return_15m is None:
        return "sector label is not mature yet"
    if directional_return_15m > 0.0 and (best_funding_support or 0.0) > 0.0:
        return "sector direction and perp funding support align"
    if directional_return_15m > 0.0:
        return "sector direction worked, but funding support is weak"
    return "sector direction failed over the current short label"


def _funding_support(*, direction: int, annualized_funding: float | None) -> float | None:
    if annualized_funding is None or direction == 0:
        return None
    return -float(direction) * annualized_funding


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float_or_none(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_category_perp_context.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_category_perp_context.md",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_category_perp_context_rows()
    write_category_perp_context_csv(rows, output_path=args.output_path)
    write_category_perp_context_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.symbol,
            row.action,
            f"dir15={_format_float(row.directional_return_15m)}",
            f"funding_support={_format_float(row.best_funding_support)}",
            f"score={row.context_score:.4f}",
        )


if __name__ == "__main__":
    main()
