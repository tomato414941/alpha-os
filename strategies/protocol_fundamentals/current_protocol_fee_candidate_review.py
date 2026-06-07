from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class ProtocolFeeCandidateReviewRow:
    token_symbol: str
    protocol: str
    fee_status: str
    side: str
    fee_score: float
    fee_growth_7d: float
    funding: float
    sector_directional_return_15m: float
    protocol_activity_directional_return_15m: float
    unlock_status: str
    unlock_side: str
    perp_pressure_action: str
    review_status: str
    score: float
    evidence: str
    next_step: str


def build_protocol_fee_candidate_review_rows(
    *,
    fee_path: Path = ROOT / "current_protocol_fee_screen.csv",
    sector_label_path: Path = STRATEGIES_ROOT / "sector_rotation" / "current_category_tradable_forward_labels.csv",
    protocol_activity_label_path: Path = STRATEGIES_ROOT
    / "protocol_activity"
    / "current_protocol_activity_forward_labels.csv",
    unlock_path: Path = STRATEGIES_ROOT / "token_unlocks" / "current_token_unlock_paper_tickets.csv",
    okx_pressure_label_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_okx_perp_pressure_forward_labels.csv",
) -> tuple[ProtocolFeeCandidateReviewRow, ...]:
    sector_labels = _best_label_by_symbol(sector_label_path, symbol_key="symbol")
    activity_labels = _best_label_by_symbol(protocol_activity_label_path, symbol_key="symbol")
    unlocks = {row.get("symbol", ""): row for row in _read_rows(unlock_path)}
    pressure = {row.get("asset", ""): row for row in _read_rows(okx_pressure_label_path)}
    rows = []
    for fee in _read_rows(fee_path):
        if fee.get("status") not in {"paper_long_context", "funding_crowded_watch"}:
            continue
        token = fee.get("token_symbol", "")
        rows.append(
            _build_row(
                fee=fee,
                sector_label=sector_labels.get(token, {}),
                activity_label=activity_labels.get(token, {}),
                unlock=unlocks.get(token, {}),
                pressure=pressure.get(token, {}),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_protocol_fee_candidate_review_csv(
    rows: tuple[ProtocolFeeCandidateReviewRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "token_symbol",
                "protocol",
                "fee_status",
                "side",
                "fee_score",
                "fee_growth_7d",
                "funding",
                "sector_directional_return_15m",
                "protocol_activity_directional_return_15m",
                "unlock_status",
                "unlock_side",
                "perp_pressure_action",
                "review_status",
                "score",
                "evidence",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.token_symbol,
                    row.protocol,
                    row.fee_status,
                    row.side,
                    f"{row.fee_score:.8f}",
                    f"{row.fee_growth_7d:.8f}",
                    f"{row.funding:.8f}",
                    f"{row.sector_directional_return_15m:.8f}",
                    f"{row.protocol_activity_directional_return_15m:.8f}",
                    row.unlock_status,
                    row.unlock_side,
                    row.perp_pressure_action,
                    row.review_status,
                    f"{row.score:.8f}",
                    row.evidence,
                    row.next_step,
                )
            )
    return output_path


def write_protocol_fee_candidate_review_md(
    rows: tuple[ProtocolFeeCandidateReviewRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Fee Candidate Review\n\n")
        handle.write(
            "This reviews protocol fee-growth candidates against sector labels, protocol activity labels, "
            "unlock conflicts, and perp-pressure context. It is not a trade instruction.\n\n"
        )
        handle.write("| token | protocol | status | score | evidence | next step |\n")
        handle.write("| --- | --- | --- | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.token_symbol} | {row.protocol} | {row.review_status} | "
                f"{row.score:.4f} | {_escape(row.evidence)} | {_escape(row.next_step)} |\n"
            )
    return output_path


def _build_row(
    *,
    fee: dict[str, str],
    sector_label: dict[str, str],
    activity_label: dict[str, str],
    unlock: dict[str, str],
    pressure: dict[str, str],
) -> ProtocolFeeCandidateReviewRow:
    token = fee.get("token_symbol", "")
    sector_return = _float(sector_label.get("directional_return_15m"))
    activity_return = _float(activity_label.get("directional_return_15m"))
    fee_score = _float(fee.get("score"))
    fee_growth = _float(fee.get("change_7d_over_7d"))
    funding = _float(fee.get("funding"))
    review_status, score, evidence = _review_status_score_evidence(
        fee=fee,
        sector_return=sector_return,
        activity_return=activity_return,
        unlock=unlock,
        pressure=pressure,
    )
    return ProtocolFeeCandidateReviewRow(
        token_symbol=token,
        protocol=fee.get("name", ""),
        fee_status=fee.get("status", ""),
        side=fee.get("side", ""),
        fee_score=fee_score,
        fee_growth_7d=fee_growth,
        funding=funding,
        sector_directional_return_15m=sector_return,
        protocol_activity_directional_return_15m=activity_return,
        unlock_status=unlock.get("status", ""),
        unlock_side=unlock.get("side", ""),
        perp_pressure_action=pressure.get("action", ""),
        review_status=review_status,
        score=score,
        evidence=evidence,
        next_step=_next_step(token=token, review_status=review_status),
    )


def _review_status_score_evidence(
    *,
    fee: dict[str, str],
    sector_return: float,
    activity_return: float,
    unlock: dict[str, str],
    pressure: dict[str, str],
) -> tuple[str, float, str]:
    fee_score = _float(fee.get("score"))
    fee_growth = _float(fee.get("change_7d_over_7d"))
    funding = _float(fee.get("funding"))
    score = fee_score + max(fee_growth, 0.0) / 20.0
    evidence_parts = [
        f"fee_growth_7d={fee_growth:.2f}",
        f"funding={funding:.4f}",
    ]
    if sector_return != 0.0:
        score += sector_return * 1000.0
        evidence_parts.append(f"sector15={sector_return:.6f}")
    if activity_return != 0.0:
        score += activity_return * 1000.0
        evidence_parts.append(f"activity15={activity_return:.6f}")
    if pressure.get("action"):
        evidence_parts.append(f"perp_pressure={pressure.get('action', '')}")
    if unlock:
        evidence_parts.append(f"unlock={unlock.get('status', '')}/{unlock.get('side', '')}")
        if unlock.get("side") == "short":
            score -= 12.0
            return "fee_growth_unlock_conflict", score, "; ".join(evidence_parts)
        if unlock.get("status") == "crowded_short_risk":
            score += 5.0
            return "fee_growth_squeeze_context", score, "; ".join(evidence_parts)
    if sector_return > 0.0 or activity_return > 0.0:
        return "fee_growth_supported_watch", score, "; ".join(evidence_parts)
    if sector_return < 0.0 or activity_return < 0.0:
        return "fee_growth_early_or_lagging", score, "; ".join(evidence_parts)
    return "fee_growth_unconfirmed", score, "; ".join(evidence_parts)


def _next_step(*, token: str, review_status: str) -> str:
    if review_status == "fee_growth_unlock_conflict":
        return f"separate {token} protocol growth thesis from unlock short pressure and label both windows"
    if review_status == "fee_growth_supported_watch":
        return f"repeat {token} fee-growth label with sector/activity support over 4h, 12h, and 24h"
    if review_status == "fee_growth_early_or_lagging":
        return f"test whether {token} fee growth leads price after short-term negative labels"
    if review_status == "fee_growth_squeeze_context":
        return f"check whether {token} crowded-short context turns fee growth into squeeze risk"
    return f"collect another {token} fee-growth snapshot and label forward returns"


def _best_label_by_symbol(path: Path, *, symbol_key: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        symbol = row.get(symbol_key, "")
        if not symbol:
            continue
        current = output.get(symbol)
        if current is None or abs(_float(row.get("directional_return_15m"))) > abs(
            _float(current.get("directional_return_15m"))
        ):
            output[symbol] = row
    return output


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
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_protocol_fee_candidate_review.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_candidate_review.md",
    )
    args = parser.parse_args()

    rows = build_protocol_fee_candidate_review_rows()
    write_protocol_fee_candidate_review_csv(rows, output_path=args.output_path)
    write_protocol_fee_candidate_review_md(rows, output_path=args.markdown_output_path)
    for row in rows[:10]:
        print(row.review_status, row.token_symbol, f"score={row.score:.4f}", row.evidence)


if __name__ == "__main__":
    main()
