from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AlphaStackRow:
    opportunity: str
    status: str
    side: str
    priority_score: float
    sources: str
    evidence: str
    conflict: str
    next_step: str


def build_alpha_stack(root: Path = ROOT) -> tuple[AlphaStackRow, ...]:
    rows = [
        _btc_risk_off_short_stack(root),
        _mstr_btc_relative_value_stack(root),
        _btc_options_volatility_stack(root),
        _prediction_market_event_model_stack(root),
        *_token_unlock_stacks(root),
        _liquidation_flow_stack(root),
        _l2_imbalance_stack(root),
    ]
    return tuple(
        sorted((row for row in rows if row is not None), key=lambda row: row.priority_score, reverse=True)
    )


def write_alpha_stack_csv(rows: tuple[AlphaStackRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("opportunity", "status", "side", "priority_score", "sources", "evidence", "conflict", "next_step")
        )
        for row in rows:
            writer.writerow(
                (
                    row.opportunity,
                    row.status,
                    row.side,
                    f"{row.priority_score:.8f}",
                    row.sources,
                    row.evidence,
                    row.conflict,
                    row.next_step,
                )
            )
    return output_path


def write_alpha_stack_md(rows: tuple[AlphaStackRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Alpha Stack\n\n")
        handle.write(
            "This stack joins current paper tickets and watches across lanes. "
            "It is a candidate-generation view, not an approval list or trade instruction.\n\n"
        )
        handle.write("| opportunity | status | side | priority score | sources | evidence | conflict | next step |\n")
        handle.write("| --- | --- | --- | ---: | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.opportunity} | {row.status} | {row.side} | {row.priority_score:.4f} | "
                f"{row.sources} | {_escape(row.evidence)} | {_escape(row.conflict)} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _btc_risk_off_short_stack(root: Path) -> AlphaStackRow | None:
    institutional = _best_by_score(
        root / "institutional_flow" / "current_btc_etf_funding_paper_ticket.csv",
        score_key="score",
        status_values={"paper_venue_candidate"},
    )
    macro = _row_by_name(root / "macro_regime" / "current_macro_crypto_paper_tickets.csv", "crypto_risk_off_lagged_short")
    speculative = _row_by_name(
        root / "speculative_beta" / "current_speculative_beta_paper_tickets.csv",
        "vix_high_beta_air_pocket",
    )
    if not institutional:
        return None
    raw_score = _float(institutional.get("score")) + _abs_float(macro.get("score") if macro else "") + _abs_float(
        speculative.get("score") if speculative else ""
    )
    evidence_parts = [
        f"{institutional.get('venue', '')}/{institutional.get('instrument', '')} {institutional.get('side', '')} funding ticket score={institutional.get('score', '')}",
    ]
    if macro:
        evidence_parts.append(macro.get("reason", ""))
    if speculative:
        evidence_parts.append(speculative.get("reason", ""))
    return AlphaStackRow(
        opportunity="btc_risk_off_short_stack",
        status="paper_watch",
        side="short_btc_perp",
        priority_score=_priority_score("paper_watch", source_count=3, raw_score=raw_score),
        sources="institutional_flow + macro_regime + speculative_beta",
        evidence="; ".join(part for part in evidence_parts if part),
        conflict="BTC and ETH may already have repriced lower; Deribit put-skew screen points to rich downside vol rather than clean directional short",
        next_step="label 4h/12h/24h BTC outcomes when ETF/funding short watch overlaps macro and speculative-beta risk-off pressure",
    )


def _mstr_btc_relative_value_stack(root: Path) -> AlphaStackRow | None:
    mstr = _row_by_name(
        root / "crypto_equity_proxy" / "current_crypto_equity_proxy_paper_tickets.csv",
        "mstr_btc_dislocation",
    )
    prediction = _first_matching(
        root / "prediction_markets" / "current_prediction_market_paper_tickets.csv",
        lambda row: "Microstrategy" in row.get("question", "") and row.get("status") == "paper_event_model_candidate",
    )
    if not mstr:
        return None
    raw_score = _abs_float(mstr.get("score")) * 100.0
    if prediction:
        raw_score += min(_float(prediction.get("score")), 100.0) / 10.0
    evidence = mstr.get("reason", "")
    if prediction:
        evidence = f"{evidence}; prediction market event model candidate: {prediction.get('question', '')} {prediction.get('outcome', '')}"
    return AlphaStackRow(
        opportunity="mstr_btc_relative_value",
        status=mstr.get("status", "paper_relative_value_watch"),
        side=mstr.get("side", "long_mstr_short_btc"),
        priority_score=_priority_score(mstr.get("status", ""), source_count=2, raw_score=raw_score),
        sources="crypto_equity_proxy + prediction_markets",
        evidence=evidence,
        conflict="requires equity borrow/liquidity, corporate-news check, and BTC hedge mapping; prediction market odds are not a direct equity fair-value model",
        next_step="label MSTR/BTC relative returns around BTC-purchase news and compare against borrow, spread, and hedge slippage",
    )


def _btc_options_volatility_stack(root: Path) -> AlphaStackRow | None:
    ticket = _best_by_score(
        root / "options_volatility" / "current_options_volatility_paper_tickets.csv",
        score_key="score",
        status_values={"paper_short_put_spread_candidate"},
    )
    if not ticket:
        return None
    return AlphaStackRow(
        opportunity="btc_options_short_put_spread",
        status=ticket.get("status", "paper_short_put_spread_candidate"),
        side=f"{ticket.get('currency', 'BTC')}_{ticket.get('structure', 'short_put_spread')}",
        priority_score=_priority_score(ticket.get("status", ""), source_count=1, raw_score=_float(ticket.get("score"))),
        sources="options_volatility",
        evidence=(
            f"{ticket.get('currency', '')} {ticket.get('expiry', '')}: "
            f"iv_premium_24h={ticket.get('iv_premium_24h', '')}, "
            f"skew={ticket.get('skew_iv', '')}, volume_usd={ticket.get('volume_usd', '')}"
        ),
        conflict="macro/speculative-beta risk-off pressure can turn rich put premium into real tail loss",
        next_step="paper-check bid/ask spread, margin, max loss, delta hedge cost, and behavior during the current VIX shock",
    )


def _prediction_market_event_model_stack(root: Path) -> AlphaStackRow | None:
    ticket = _best_by_score(
        root / "prediction_markets" / "current_prediction_market_paper_tickets.csv",
        score_key="score",
        status_values={"paper_event_model_candidate", "paper_event_model_watch"},
    )
    if not ticket:
        return None
    return AlphaStackRow(
        opportunity="prediction_market_event_model",
        status=ticket.get("status", "paper_event_model_candidate"),
        side=f"{ticket.get('question', '')} {ticket.get('outcome', '')}",
        priority_score=_priority_score(ticket.get("status", ""), source_count=1, raw_score=_float(ticket.get("score"))),
        sources="prediction_markets",
        evidence=(
            f"{ticket.get('category', '')}: spread={ticket.get('spread', '')}, "
            f"depth={ticket.get('visible_depth_score', '')}, vol24={ticket.get('volume_24h', '')}"
        ),
        conflict="market depth is not edge; needs independent true-probability model and latency/adverse-selection checks",
        next_step="build external news/filing probability model before any paper event-market action",
    )


def _token_unlock_stacks(root: Path) -> tuple[AlphaStackRow, ...]:
    rows = _read_rows(root / "token_unlocks" / "current_token_unlock_paper_tickets.csv")
    tickets = sorted(
        (row for row in rows if row.get("status") in {"paper_short_candidate", "crowded_short_risk"}),
        key=lambda row: _float(row.get("score")),
        reverse=True,
    )
    output: list[AlphaStackRow] = []
    for ticket in tickets[:3]:
        symbol = ticket.get("symbol", "")
        status = ticket.get("status", "")
        side = ticket.get("side", "")
        output.append(
            AlphaStackRow(
                opportunity=f"{symbol.lower()}_unlock_event",
                status=status,
                side=side,
                priority_score=_priority_score(status, source_count=1, raw_score=_float(ticket.get("score"))),
                sources="token_unlocks",
                evidence=(
                    f"{symbol}: unlock_value={ticket.get('unlock_value_usd', '')}, "
                    f"supply={ticket.get('percent_supply', '')}%, funding={ticket.get('annualized_funding', '')}, "
                    f"impact={ticket.get('impact_spread', '')}"
                ),
                conflict="unlock event can be crowded or already priced; negative funding can turn short into squeeze risk",
                next_step=f"label {symbol} pre/post unlock returns, funding persistence, depth decay, and stop behavior",
            )
        )
    return tuple(output)


def _liquidation_flow_stack(root: Path) -> AlphaStackRow | None:
    ticket = _best_by_score(
        root / "liquidation_flow" / "current_okx_liquidation_paper_gate.csv",
        score_key="conservative_net_bps",
        status_values={"small_paper_probe"},
        status_key="gate_action",
    )
    if not ticket:
        return None
    asset = ticket.get("asset", "")
    return AlphaStackRow(
        opportunity=f"{asset.lower()}_liquidation_continuation",
        status=ticket.get("gate_action", "small_paper_probe"),
        side=ticket.get("action", ""),
        priority_score=_priority_score(
            ticket.get("gate_action", ""),
            source_count=1,
            raw_score=_float(ticket.get("conservative_net_bps")),
        ),
        sources="liquidation_flow",
        evidence=(
            f"{asset}: net15={ticket.get('conservative_net_bps', '')}bps, "
            f"size={ticket.get('candidate_size_usd', '')}, depth_usage={ticket.get('visible_depth_usage', '')}"
        ),
        conflict="retrospective paper outcome can overstate edge; needs fresh-event repeats and live depth/fill checks",
        next_step=f"repeat {asset} liquidation event on fresh observations with fees, spread, fill, and funding included",
    )


def _l2_imbalance_stack(root: Path) -> AlphaStackRow | None:
    ticket = _best_by_score(
        root / "market_making" / "current_l2_imbalance_paper_gate.csv",
        score_key="net_15m_bps",
        status_values={"small_paper_probe"},
        status_key="gate_action",
    )
    if not ticket:
        return None
    asset = ticket.get("asset", "")
    return AlphaStackRow(
        opportunity=f"{asset.lower()}_l2_imbalance_probe",
        status=ticket.get("gate_action", "small_paper_probe"),
        side="directional_l2_probe",
        priority_score=_priority_score(
            ticket.get("gate_action", ""),
            source_count=1,
            raw_score=_float(ticket.get("net_15m_bps")),
        ),
        sources="market_making",
        evidence=(
            f"{asset}: net15={ticket.get('net_15m_bps', '')}bps, "
            f"imbalance_10bps={ticket.get('imbalance_10_bps', '')}, "
            f"depth_usage={ticket.get('visible_depth_usage', '')}"
        ),
        conflict="directional L2 probe is not maker edge; queue position, fill probability, and adverse selection are still missing",
        next_step=f"collect repeated {asset} L2 snapshots with trade prints and estimate fill-side next return",
    )


def _best_by_score(
    path: Path,
    *,
    score_key: str,
    status_values: set[str],
    status_key: str = "status",
) -> dict[str, str] | None:
    rows = tuple(row for row in _read_rows(path) if row.get(status_key) in status_values)
    if not rows:
        return None
    return max(rows, key=lambda row: _float(row.get(score_key)))


def _row_by_name(path: Path, name: str) -> dict[str, str] | None:
    return _first_matching(path, lambda row: row.get("name") == name)


def _first_matching(path: Path, predicate: object) -> dict[str, str] | None:
    for row in _read_rows(path):
        if predicate(row):
            return row
    return None


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _abs_float(value: str | None) -> float:
    return abs(_float(value))


def _priority_score(status: str, *, source_count: int, raw_score: float) -> float:
    status_score = {
        "paper_event_model_candidate": 75.0,
        "paper_short_candidate": 72.0,
        "paper_long_candidate": 72.0,
        "paper_short_put_spread_candidate": 68.0,
        "paper_relative_value_watch": 64.0,
        "small_paper_probe": 60.0,
        "paper_watch": 52.0,
        "crowded_short_risk": 48.0,
        "paper_risk_context": 45.0,
        "watch": 35.0,
    }.get(status, 30.0)
    return status_score + min(source_count * 7.5, 25.0) + min(abs(raw_score) / 10.0, 15.0)


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_alpha_stack.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_alpha_stack.md")
    args = parser.parse_args()

    rows = build_alpha_stack()
    write_alpha_stack_csv(rows, output_path=args.output_path)
    write_alpha_stack_md(rows, output_path=args.markdown_output_path)
    for row in rows[:10]:
        print(row.status, row.side, f"priority={row.priority_score:.4f}", row.opportunity)


if __name__ == "__main__":
    main()
