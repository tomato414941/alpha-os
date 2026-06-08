from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EventCryptoHedgeSurvival:
    candidate_id: str
    market_id: str
    question: str
    asset: str
    hedge_action: str
    event_bias: str
    survival_status: str
    survival_score: float
    candidate_status: str
    reaction_status: str
    attribution_status: str
    alignment_status: str
    probability_gap: float
    current_edge_after_ask: float
    asset_directional_return_bps: float
    basket_directional_return_bps: float
    residual_vs_basket_bps: float
    event_mark_return_bps: float
    same_asset_control_gap_bps: float
    reason: str
    next_step: str


def build_event_crypto_hedge_survival_rows(
    *,
    candidates_path: Path = ROOT / "current_event_crypto_hedge_candidates.csv",
    reactions_path: Path = ROOT / "current_event_crypto_hedge_reaction_labels.csv",
    attributions_path: Path = ROOT / "current_event_crypto_hedge_beta_attribution.csv",
    alignments_path: Path = ROOT / "current_event_crypto_hedge_event_alignment.csv",
) -> tuple[EventCryptoHedgeSurvival, ...]:
    reactions = {row.get("candidate_id", ""): row for row in _read_rows(reactions_path)}
    attributions = {row.get("candidate_id", ""): row for row in _read_rows(attributions_path)}
    alignments = {row.get("candidate_id", ""): row for row in _read_rows(alignments_path)}
    rows = tuple(
        _build_row(
            candidate=row,
            reaction=reactions.get(row.get("candidate_id", ""), {}),
            attribution=attributions.get(row.get("candidate_id", ""), {}),
            alignment=alignments.get(row.get("candidate_id", ""), {}),
        )
        for row in _read_rows(candidates_path)
    )
    return tuple(sorted(rows, key=lambda row: row.survival_score, reverse=True))


def write_event_crypto_hedge_survival_csv(
    rows: tuple[EventCryptoHedgeSurvival, ...],
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
                "survival_status",
                "survival_score",
                "candidate_status",
                "reaction_status",
                "attribution_status",
                "alignment_status",
                "probability_gap",
                "current_edge_after_ask",
                "asset_directional_return_bps",
                "basket_directional_return_bps",
                "residual_vs_basket_bps",
                "event_mark_return_bps",
                "same_asset_control_gap_bps",
                "reason",
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
                    row.survival_status,
                    f"{row.survival_score:.8f}",
                    row.candidate_status,
                    row.reaction_status,
                    row.attribution_status,
                    row.alignment_status,
                    f"{row.probability_gap:.6f}",
                    f"{row.current_edge_after_ask:.6f}",
                    f"{row.asset_directional_return_bps:.8f}",
                    f"{row.basket_directional_return_bps:.8f}",
                    f"{row.residual_vs_basket_bps:.8f}",
                    f"{row.event_mark_return_bps:.8f}",
                    f"{row.same_asset_control_gap_bps:.8f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_event_crypto_hedge_survival_md(
    rows: tuple[EventCryptoHedgeSurvival, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Event Crypto Hedge Survival\n\n")
        handle.write(
            "This checks whether a prediction-market-derived crypto hedge survives reaction, beta, "
            "same-asset control, and event-price alignment checks. It is not a live trade instruction.\n\n"
        )
        handle.write(
            "| candidate | status | score | asset bps | event bps | basket bps | residual | control gap | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.survival_status} | "
                f"{row.survival_score:.4f} | "
                f"{row.asset_directional_return_bps:.4f} | "
                f"{row.event_mark_return_bps:.4f} | "
                f"{row.basket_directional_return_bps:.4f} | "
                f"{row.residual_vs_basket_bps:.4f} | "
                f"{row.same_asset_control_gap_bps:.4f} | "
                f"{_escape(row.reason)} |\n"
            )
    return output_path


def _build_row(
    *,
    candidate: dict[str, str],
    reaction: dict[str, str],
    attribution: dict[str, str],
    alignment: dict[str, str],
) -> EventCryptoHedgeSurvival:
    status = _survival_status(
        candidate_status=candidate.get("status", ""),
        reaction_status=reaction.get("reaction_status", ""),
        attribution_status=attribution.get("attribution_status", ""),
        alignment_status=alignment.get("alignment_status", ""),
    )
    asset_return = _float(alignment.get("asset_directional_return_bps") or attribution.get("asset_directional_return_bps"))
    event_return = _float(alignment.get("event_mark_return_bps"))
    basket_return = _float(attribution.get("basket_directional_return_bps"))
    residual = _float(alignment.get("residual_vs_basket_bps") or attribution.get("residual_vs_basket_bps"))
    control_gap = _float(alignment.get("same_asset_control_gap_bps"))
    probability_gap = _float(candidate.get("probability_gap"))
    current_edge = _float(candidate.get("current_edge_after_ask"))
    return EventCryptoHedgeSurvival(
        candidate_id=candidate.get("candidate_id", ""),
        market_id=candidate.get("market_id", ""),
        question=candidate.get("question", ""),
        asset=candidate.get("asset", ""),
        hedge_action=candidate.get("hedge_action", ""),
        event_bias=candidate.get("event_bias", ""),
        survival_status=status,
        survival_score=_survival_score(
            status=status,
            probability_gap=probability_gap,
            current_edge_after_ask=current_edge,
            asset_directional_return_bps=asset_return,
            event_mark_return_bps=event_return,
            residual_vs_basket_bps=residual,
            same_asset_control_gap_bps=control_gap,
        ),
        candidate_status=candidate.get("status", ""),
        reaction_status=reaction.get("reaction_status", ""),
        attribution_status=attribution.get("attribution_status", ""),
        alignment_status=alignment.get("alignment_status", ""),
        probability_gap=probability_gap,
        current_edge_after_ask=current_edge,
        asset_directional_return_bps=asset_return,
        basket_directional_return_bps=basket_return,
        residual_vs_basket_bps=residual,
        event_mark_return_bps=event_return,
        same_asset_control_gap_bps=control_gap,
        reason=_reason(status),
        next_step=_next_step(status),
    )


def _survival_status(
    *,
    candidate_status: str,
    reaction_status: str,
    attribution_status: str,
    alignment_status: str,
) -> str:
    if alignment_status == "event_probability_and_crypto_aligned":
        return "event_crypto_hedge_survived_alignment"
    if alignment_status == "event_probability_flat_crypto_moved":
        return "event_crypto_hedge_rejected_event_flat"
    if alignment_status == "same_asset_control_explains_return":
        return "event_crypto_hedge_rejected_same_asset_control"
    if alignment_status == "event_probability_contradicts_crypto":
        return "event_crypto_hedge_rejected_event_contradiction"
    if attribution_status == "event_crypto_residual_outperformance":
        return "event_crypto_hedge_residual_watch"
    if reaction_status == "event_crypto_hedge_reaction_win":
        return "event_crypto_hedge_beta_context_only"
    if reaction_status == "event_crypto_hedge_reaction_pending":
        return "event_crypto_hedge_pending_mark"
    if candidate_status:
        return "event_crypto_hedge_candidate_unproven"
    return "event_crypto_hedge_missing_evidence"


def _survival_score(
    *,
    status: str,
    probability_gap: float,
    current_edge_after_ask: float,
    asset_directional_return_bps: float,
    event_mark_return_bps: float,
    residual_vs_basket_bps: float,
    same_asset_control_gap_bps: float,
) -> float:
    base = {
        "event_crypto_hedge_survived_alignment": 120.0,
        "event_crypto_hedge_residual_watch": 65.0,
        "event_crypto_hedge_pending_mark": 30.0,
        "event_crypto_hedge_candidate_unproven": 20.0,
        "event_crypto_hedge_beta_context_only": 5.0,
        "event_crypto_hedge_rejected_same_asset_control": -50.0,
        "event_crypto_hedge_rejected_event_flat": -70.0,
        "event_crypto_hedge_rejected_event_contradiction": -90.0,
    }.get(status, 0.0)
    return (
        base
        + probability_gap * 50.0
        + current_edge_after_ask * 30.0
        + max(event_mark_return_bps, 0.0) * 0.8
        + max(residual_vs_basket_bps, 0.0) * 0.15
        + max(same_asset_control_gap_bps, 0.0) * 0.05
        + max(asset_directional_return_bps, 0.0) * 0.03
    )


def _reason(status: str) -> str:
    if status == "event_crypto_hedge_survived_alignment":
        return "event market and crypto hedge moved in the same direction after controls"
    if status == "event_crypto_hedge_rejected_event_flat":
        return "crypto moved, but the event-market paper mark was flat"
    if status == "event_crypto_hedge_rejected_same_asset_control":
        return "same-asset non-event controls explain the return"
    if status == "event_crypto_hedge_rejected_event_contradiction":
        return "event-market movement contradicted the crypto hedge"
    if status == "event_crypto_hedge_beta_context_only":
        return "crypto reaction won, but evidence is common beta rather than event alpha"
    if status == "event_crypto_hedge_residual_watch":
        return "asset residual is interesting, but still needs event alignment"
    if status == "event_crypto_hedge_pending_mark":
        return "candidate is waiting for a usable reaction mark"
    return "candidate does not yet have enough survival evidence"


def _next_step(status: str) -> str:
    if status == "event_crypto_hedge_survived_alignment":
        return "repeat on fresh event markets with explicit funding, spread/depth, and timestamp controls"
    if status in {
        "event_crypto_hedge_rejected_event_flat",
        "event_crypto_hedge_rejected_same_asset_control",
        "event_crypto_hedge_rejected_event_contradiction",
    }:
        return "do not promote; require event-price movement before treating the crypto move as event alpha"
    if status == "event_crypto_hedge_pending_mark":
        return "wait for the mark, then rerun beta and event-alignment checks"
    return "keep as context until event alignment and controls survive"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_event_crypto_hedge_survival.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_event_crypto_hedge_survival.md")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_event_crypto_hedge_survival_rows()
    write_event_crypto_hedge_survival_csv(rows, output_path=args.output_path)
    write_event_crypto_hedge_survival_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.survival_status, row.candidate_id, f"{row.survival_score:.4f}")


if __name__ == "__main__":
    main()
