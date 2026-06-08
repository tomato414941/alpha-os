from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VolatilityHedgeCandidate:
    candidate_id: str
    currency: str
    expiry: str
    structure: str
    hedge_profile: str
    decision: str
    score: float
    days_to_expiry: float
    iv_premium_24h: float
    max_loss_pct: float
    max_loss_usd: float
    realized_move_pct: float
    premium_to_realized_move: float
    quote_spread_pct: float
    top_ask_premium_depth_usd: float
    hedge_interval: str
    reason: str
    next_step: str


def build_volatility_hedge_candidates(root: Path = ROOT) -> tuple[VolatilityHedgeCandidate, ...]:
    actionability_by_key = {
        (row.get("currency", ""), row.get("expiry", ""), row.get("structure", "")): row
        for row in _read_rows(root / "current_volatility_actionability.csv")
    }
    output: list[VolatilityHedgeCandidate] = []
    for row in _read_rows(root / "current_options_volatility_paper_tickets.csv"):
        key = (row.get("currency", ""), row.get("expiry", ""), row.get("structure", ""))
        actionability = actionability_by_key.get(key, {})
        decision = _decision(actionability.get("status", row.get("status", "")))
        if decision == "ignore_volatility_hedge":
            continue
        dte = _float(row.get("days_to_expiry"))
        hedge_profile, hedge_interval = _hedge_profile(dte=dte)
        score = _score(
            decision=decision,
            actionability_score=_float(actionability.get("score")),
            max_loss_pct=_float(row.get("max_loss_pct")),
            premium_to_realized_move=_float(row.get("premium_to_realized_move")),
            quote_spread_pct=_float(row.get("quote_spread_pct")),
            top_ask_premium_depth_usd=_float(row.get("top_ask_premium_depth_usd")),
        )
        output.append(
            VolatilityHedgeCandidate(
                candidate_id=f"{row.get('currency', '').lower()}_{row.get('expiry', '').replace('-', '')}_{hedge_profile}",
                currency=row.get("currency", ""),
                expiry=row.get("expiry", ""),
                structure=row.get("structure", ""),
                hedge_profile=hedge_profile,
                decision=decision,
                score=score,
                days_to_expiry=dte,
                iv_premium_24h=_float(row.get("iv_premium_24h")),
                max_loss_pct=_float(row.get("max_loss_pct")),
                max_loss_usd=_float(row.get("max_loss_usd")),
                realized_move_pct=_float(row.get("realized_move_pct")),
                premium_to_realized_move=_float(row.get("premium_to_realized_move")),
                quote_spread_pct=_float(row.get("quote_spread_pct")),
                top_ask_premium_depth_usd=_float(row.get("top_ask_premium_depth_usd")),
                hedge_interval=hedge_interval,
                reason=_reason(decision=decision, hedge_profile=hedge_profile),
                next_step=_next_step(
                    currency=row.get("currency", ""),
                    expiry=row.get("expiry", ""),
                    hedge_profile=hedge_profile,
                    decision=decision,
                ),
            )
        )
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_volatility_hedge_candidates_csv(
    rows: tuple[VolatilityHedgeCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "currency",
                "expiry",
                "structure",
                "hedge_profile",
                "decision",
                "score",
                "days_to_expiry",
                "iv_premium_24h",
                "max_loss_pct",
                "max_loss_usd",
                "realized_move_pct",
                "premium_to_realized_move",
                "quote_spread_pct",
                "top_ask_premium_depth_usd",
                "hedge_interval",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.currency,
                    row.expiry,
                    row.structure,
                    row.hedge_profile,
                    row.decision,
                    f"{row.score:.8f}",
                    f"{row.days_to_expiry:.4f}",
                    f"{row.iv_premium_24h:.4f}",
                    f"{row.max_loss_pct:.6f}",
                    f"{row.max_loss_usd:.2f}",
                    f"{row.realized_move_pct:.6f}",
                    f"{row.premium_to_realized_move:.6f}",
                    f"{row.quote_spread_pct:.6f}",
                    f"{row.top_ask_premium_depth_usd:.2f}",
                    row.hedge_interval,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_volatility_hedge_candidates_md(
    rows: tuple[VolatilityHedgeCandidate, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Volatility Hedge Candidates\n\n")
        handle.write(
            "This turns option actionability rows into hedge-plan candidates. "
            "It is not a live options execution instruction.\n\n"
        )
        handle.write(
            "| candidate | decision | score | max loss % | max loss USD | prem/rv move | spread | depth USD | hedge interval | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.decision} | "
                f"{row.score:.4f} | "
                f"{row.max_loss_pct:.2f} | "
                f"{row.max_loss_usd:.2f} | "
                f"{row.premium_to_realized_move:.2f} | "
                f"{row.quote_spread_pct:.4f} | "
                f"{row.top_ask_premium_depth_usd:.0f} | "
                f"{row.hedge_interval} | "
                f"{_escape(row.reason)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Long-vol alpha is only useful if the option quote, premium-at-risk, delta hedge path, "
            "exit bid, and margin treatment are explicit. Rows here identify which structures are "
            "ready for that paper hedge check.\n"
        )
    return output_path


def _decision(status: str) -> str:
    if status == "volatility_candidate_needs_sweep_hedge":
        return "paper_delta_hedge_candidate"
    if status == "volatility_short_expiry_hedge_watch":
        return "expiry_gamma_hedge_watch"
    if status == "volatility_quote_mechanics_watch":
        return "quote_only_hedge_watch"
    return "ignore_volatility_hedge"


def _hedge_profile(*, dte: float) -> tuple[str, str]:
    if dte < 3.0:
        return "expiry_gamma_scalp_check", "2h_or_large_delta_move"
    if dte <= 14.0:
        return "short_dated_delta_hedge_check", "4h_or_large_delta_move"
    return "medium_dated_delta_hedge_check", "daily_or_large_delta_move"


def _score(
    *,
    decision: str,
    actionability_score: float,
    max_loss_pct: float,
    premium_to_realized_move: float,
    quote_spread_pct: float,
    top_ask_premium_depth_usd: float,
) -> float:
    decision_base = {
        "paper_delta_hedge_candidate": 45.0,
        "expiry_gamma_hedge_watch": 34.0,
        "quote_only_hedge_watch": 30.0,
    }.get(decision, 0.0)
    return (
        decision_base
        + min(actionability_score / 3.0, 24.0)
        + min(top_ask_premium_depth_usd / 50_000.0, 8.0)
        - max(max_loss_pct - 10.0, 0.0) * 0.8
        - max(premium_to_realized_move - 0.55, 0.0) * 20.0
        - max(quote_spread_pct - 0.04, 0.0) * 120.0
    )


def _reason(*, decision: str, hedge_profile: str) -> str:
    if decision == "paper_delta_hedge_candidate":
        return f"quote and premium are good enough to test {hedge_profile} with explicit hedge PnL"
    if decision == "expiry_gamma_hedge_watch":
        return "near-expiry gamma may be useful, but hedge timing dominates the result"
    if decision == "quote_only_hedge_watch":
        return "quote exists, but depth or mechanics are not strong enough for hedge promotion"
    return "not a hedge candidate"


def _next_step(*, currency: str, expiry: str, hedge_profile: str, decision: str) -> str:
    subject = f"{currency} {expiry} {hedge_profile}"
    if decision == "paper_delta_hedge_candidate":
        return f"paper-check {subject}: sweep depth, delta hedge marks, exit bid, max loss, margin, and stop"
    if decision == "expiry_gamma_hedge_watch":
        return f"paper-check {subject}: event timing, gamma scalping interval, exit bid, and expiry handling"
    return f"keep {subject} at quote-mechanics review until sweep depth and hedge path improve"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_volatility_hedge_candidates.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_volatility_hedge_candidates.md")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    rows = build_volatility_hedge_candidates()
    write_volatility_hedge_candidates_csv(rows, output_path=args.output_path)
    write_volatility_hedge_candidates_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.candidate_id, row.decision, f"{row.score:.4f}", row.next_step)


if __name__ == "__main__":
    main()
