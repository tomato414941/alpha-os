from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventCryptoHedgeBetaAttribution:
    candidate_id: str
    market_id: str
    question: str
    asset: str
    hedge_action: str
    event_bias: str
    reaction_status: str
    checkpoint_status: str
    elapsed_minutes: str
    asset_directional_return_bps: str
    basket_directional_return_bps: str
    residual_vs_basket_bps: str
    attribution_status: str
    probability_gap: str
    current_edge_after_ask: str
    next_step: str


def build_event_crypto_hedge_beta_attribution(
    *,
    reaction_labels_path: Path = ROOT / "current_event_crypto_hedge_reaction_labels.csv",
) -> tuple[EventCryptoHedgeBetaAttribution, ...]:
    reaction_rows = _read_rows(reaction_labels_path)
    baskets = _basket_returns_by_market(reaction_rows)
    rows = [
        _build_attribution(row=row, basket_return=baskets.get(row.get("market_id", ""), 0.0))
        for row in reaction_rows
    ]
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_event_crypto_hedge_beta_attribution_csv(
    rows: tuple[EventCryptoHedgeBetaAttribution, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "market_id",
                "question",
                "asset",
                "hedge_action",
                "event_bias",
                "reaction_status",
                "checkpoint_status",
                "elapsed_minutes",
                "asset_directional_return_bps",
                "basket_directional_return_bps",
                "residual_vs_basket_bps",
                "attribution_status",
                "probability_gap",
                "current_edge_after_ask",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.market_id,
                    row.question,
                    row.asset,
                    row.hedge_action,
                    row.event_bias,
                    row.reaction_status,
                    row.checkpoint_status,
                    row.elapsed_minutes,
                    row.asset_directional_return_bps,
                    row.basket_directional_return_bps,
                    row.residual_vs_basket_bps,
                    row.attribution_status,
                    row.probability_gap,
                    row.current_edge_after_ask,
                    row.next_step,
                )
            )
    return output_path


def write_event_crypto_hedge_beta_attribution_md(
    rows: tuple[EventCryptoHedgeBetaAttribution, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Crypto Hedge Beta Attribution\n\n")
        handle.write(
            "This checks whether event-crypto hedge paper returns are mostly a common BTC/ETH/SOL beta move "
            "or an asset-specific residual. It is a diagnostic, not a trade instruction.\n\n"
        )
        handle.write(
            "| candidate | asset | action | status | asset bps | basket bps | residual bps | gap | edge | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.asset} | "
                f"{row.hedge_action}_{row.event_bias} | "
                f"{row.attribution_status} | "
                f"{row.asset_directional_return_bps} | "
                f"{row.basket_directional_return_bps} | "
                f"{row.residual_vs_basket_bps} | "
                f"{row.probability_gap} | "
                f"{row.current_edge_after_ask} | "
                f"{_escape(row.next_step)} |\n"
            )
        handle.write("\n## Summary\n\n")
        handle.write(_summary_text(rows))
    return output_path


def _build_attribution(*, row: dict[str, str], basket_return: float) -> EventCryptoHedgeBetaAttribution:
    asset_return = _float(row.get("directional_return_bps"))
    residual = asset_return - basket_return if row.get("directional_return_bps") else 0.0
    status = _attribution_status(
        reaction_status=row.get("reaction_status", ""),
        asset_return=asset_return,
        basket_return=basket_return,
        residual=residual,
    )
    return EventCryptoHedgeBetaAttribution(
        candidate_id=row.get("candidate_id", ""),
        market_id=row.get("market_id", ""),
        question=row.get("question", ""),
        asset=row.get("asset", ""),
        hedge_action=row.get("hedge_action", ""),
        event_bias=row.get("event_bias", ""),
        reaction_status=row.get("reaction_status", ""),
        checkpoint_status=row.get("checkpoint_status", ""),
        elapsed_minutes=row.get("elapsed_minutes", ""),
        asset_directional_return_bps=_format_optional(row.get("directional_return_bps"), asset_return),
        basket_directional_return_bps=_format_optional(row.get("directional_return_bps"), basket_return),
        residual_vs_basket_bps=_format_optional(row.get("directional_return_bps"), residual),
        attribution_status=status,
        probability_gap=row.get("probability_gap", ""),
        current_edge_after_ask=row.get("current_edge_after_ask", ""),
        next_step=_next_step(status),
    )


def _basket_returns_by_market(rows: tuple[dict[str, str], ...]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("checkpoint_status") != "ready":
            continue
        directional_return = row.get("directional_return_bps", "")
        if not directional_return:
            continue
        grouped.setdefault(row.get("market_id", ""), []).append(_float(directional_return))
    return {
        market_id: sum(values) / len(values)
        for market_id, values in grouped.items()
        if values
    }


def _attribution_status(
    *,
    reaction_status: str,
    asset_return: float,
    basket_return: float,
    residual: float,
) -> str:
    if reaction_status == "event_crypto_hedge_reaction_pending":
        return "event_crypto_beta_attribution_pending"
    if reaction_status == "event_crypto_hedge_reaction_loss":
        return "event_crypto_beta_attribution_negative"
    if reaction_status not in {
        "event_crypto_hedge_reaction_win",
        "event_crypto_hedge_reaction_flat",
    }:
        return "event_crypto_beta_attribution_missing_mark"
    if basket_return <= 0.0:
        return "event_crypto_beta_not_supported"
    if abs(residual) >= max(40.0, abs(basket_return) * 0.75):
        if residual > 0.0 and asset_return > 0.0:
            return "event_crypto_residual_outperformance"
        return "event_crypto_residual_contradiction"
    return "event_crypto_beta_move_supported"


def _next_step(status: str) -> str:
    if status == "event_crypto_beta_move_supported":
        return "repeat on fresh event markets and add funding, spread/depth, and event timestamp controls"
    if status == "event_crypto_residual_outperformance":
        return "check whether the asset-specific residual is explained by unrelated news or liquidity"
    if status == "event_crypto_residual_contradiction":
        return "do not promote until the residual failure regime is explained"
    if status == "event_crypto_beta_attribution_pending":
        return "wait for a ready reaction label before attributing beta"
    return "refresh marks and keep this as diagnostic evidence only"


def _summary_text(rows: tuple[EventCryptoHedgeBetaAttribution, ...]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.attribution_status] = counts.get(row.attribution_status, 0) + 1
    lines = [f"- {status}: {count}" for status, count in sorted(counts.items())]
    best = max(rows, key=lambda row: _float(row.asset_directional_return_bps), default=None)
    if best:
        lines.append(
            "- best asset return: "
            f"{best.candidate_id} asset={best.asset_directional_return_bps}bps "
            f"basket={best.basket_directional_return_bps}bps residual={best.residual_vs_basket_bps}bps"
        )
    if not lines:
        lines.append("- no event crypto hedge reaction labels yet")
    return "\n".join(lines) + "\n"


def _sort_key(row: EventCryptoHedgeBetaAttribution) -> tuple[float, float]:
    status_rank = {
        "event_crypto_residual_outperformance": 4.0,
        "event_crypto_beta_move_supported": 3.0,
        "event_crypto_beta_attribution_pending": 2.0,
        "event_crypto_residual_contradiction": 1.0,
        "event_crypto_beta_attribution_negative": 0.0,
    }.get(row.attribution_status, 0.0)
    return status_rank, _float(row.asset_directional_return_bps)


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


def _format_optional(source: str | None, value: float) -> str:
    if not source:
        return ""
    return f"{value:.8f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_event_crypto_hedge_beta_attribution.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_event_crypto_hedge_beta_attribution.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_event_crypto_hedge_beta_attribution()
    write_event_crypto_hedge_beta_attribution_csv(rows, output_path=args.output_path)
    write_event_crypto_hedge_beta_attribution_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.candidate_id, row.attribution_status, row.asset_directional_return_bps)


if __name__ == "__main__":
    main()
