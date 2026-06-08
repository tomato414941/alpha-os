from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH_REFERENCE = "https://arxiv.org/abs/2601.07664"


@dataclass(frozen=True)
class CryptoEquityFactorSplitRow:
    factor_id: str
    factor_role: str
    target_asset: str
    proxy_group: str
    status: str
    side_hint: str
    score: float
    evidence: str
    missing_data: str
    next_probe: str
    research_reference: str = RESEARCH_REFERENCE


def build_crypto_equity_factor_split_rows(
    *,
    snapshot_path: Path = ROOT / "current_crypto_equity_proxy_context.csv",
    ticket_path: Path = ROOT / "current_crypto_equity_proxy_paper_tickets.csv",
) -> tuple[CryptoEquityFactorSplitRow, ...]:
    snapshots = _read_rows(snapshot_path)
    tickets = _read_rows(ticket_path)
    rows = tuple(_ticket_row(ticket) for ticket in tickets) + _market_hours_gap_rows(snapshots)
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_crypto_equity_factor_split_csv(
    rows: tuple[CryptoEquityFactorSplitRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "factor_id",
                "factor_role",
                "target_asset",
                "proxy_group",
                "status",
                "side_hint",
                "score",
                "evidence",
                "missing_data",
                "next_probe",
                "research_reference",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.factor_id,
                    row.factor_role,
                    row.target_asset,
                    row.proxy_group,
                    row.status,
                    row.side_hint,
                    f"{row.score:.8f}",
                    row.evidence,
                    row.missing_data,
                    row.next_probe,
                    row.research_reference,
                )
            )
    return output_path


def write_crypto_equity_factor_split_md(
    rows: tuple[CryptoEquityFactorSplitRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crypto Equity Factor Split\n\n")
        handle.write(
            "This separates crypto-equity proxy observations into beta-hedge, residual relative-value, "
            "stress-control, and market-hours-gap roles. It is not a trade instruction.\n\n"
        )
        handle.write("| factor | role | target | status | side | score | next probe |\n")
        handle.write("| --- | --- | --- | --- | --- | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.factor_id} | {row.factor_role} | {row.target_asset} | {row.status} | "
                f"{row.side_hint} | {row.score:.4f} | {_escape(row.next_probe)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A beta hedge candidate is different from a residual relative-value candidate. "
            "Market-hours gaps are timing controls: they should not be merged with 24/7 crypto signals "
            "until the timestamp boundary is explicit.\n"
        )
    return output_path


def _ticket_row(ticket: dict[str, str]) -> CryptoEquityFactorSplitRow:
    name = ticket.get("name", "")
    role, target, proxy_group = _factor_role_target_proxy(name, ticket.get("side", ""))
    score = abs(_float(ticket.get("score"))) * 100.0
    status = _status_for_role(role, ticket.get("status", ""))
    return CryptoEquityFactorSplitRow(
        factor_id=name,
        factor_role=role,
        target_asset=target,
        proxy_group=proxy_group,
        status=status,
        side_hint=ticket.get("side", ""),
        score=score,
        evidence=f"ticket_status={ticket.get('status', '')}; raw_score={ticket.get('score', '')}; {ticket.get('reason', '')}",
        missing_data=_missing_data(role),
        next_probe=_next_probe(role=role, target=target, factor_id=name),
    )


def _market_hours_gap_rows(snapshots: tuple[dict[str, str], ...]) -> tuple[CryptoEquityFactorSplitRow, ...]:
    if not snapshots:
        return ()
    latest_date = max(row.get("last_date", "") for row in snapshots)
    rows: list[CryptoEquityFactorSplitRow] = []
    for row in snapshots:
        if row.get("group") == "crypto" or row.get("last_date") == latest_date:
            continue
        lag_days = _date_gap(latest_date, row.get("last_date", ""))
        if lag_days <= 0:
            continue
        rows.append(
            CryptoEquityFactorSplitRow(
                factor_id=f"{row.get('symbol', '').lower()}_market_hours_gap",
                factor_role="market_hours_gap_control",
                target_asset=_target_for_group(row.get("group", "")),
                proxy_group=row.get("group", ""),
                status="timestamp_boundary_required",
                side_hint="none",
                score=min(lag_days * 25.0 + abs(_float(row.get("vs_btc_5d"))) * 100.0, 100.0),
                evidence=(
                    f"proxy_last_date={row.get('last_date', '')}; crypto_latest_date={latest_date}; "
                    f"vs_btc_5d={row.get('vs_btc_5d', '')}; vs_eth_5d={row.get('vs_eth_5d', '')}"
                ),
                missing_data="market-hours timestamp alignment, overnight crypto return split, and equity open reaction label",
                next_probe=f"separate {row.get('symbol', '')} market-hours gap before using it as crypto alpha",
            )
        )
    return tuple(rows)


def _factor_role_target_proxy(name: str, side: str) -> tuple[str, str, str]:
    if name == "mstr_btc_dislocation":
        return "residual_relative_value", "BTC", "btc_treasury_equity"
    if name == "miner_stress_vs_btc":
        return "equity_stress_control", "BTC", "miner"
    if name == "eth_treasury_proxy_lead":
        return "crypto_beta_hedge", "ETH", "eth_treasury_equity"
    if side in {"short_btc_eth", "long_btc_eth"}:
        return "crypto_beta_hedge", "BTC_ETH", "crypto_linked_equity"
    return "crypto_equity_context", "BTC_ETH", "crypto_linked_equity"


def _status_for_role(role: str, status: str) -> str:
    if role == "residual_relative_value":
        return "separate_residual_from_beta_before_label"
    if role == "equity_stress_control":
        return "stress_control_before_directional_trade"
    if role == "crypto_beta_hedge" and status in {"paper_short_candidate", "paper_long_candidate"}:
        return "beta_hedge_label_candidate"
    return "context_only"


def _missing_data(role: str) -> str:
    if role == "residual_relative_value":
        return "borrow cost, equity news, BTC beta hedge ratio, and residual return label"
    if role == "equity_stress_control":
        return "regime split, crypto beta attribution, and negative-control outcome"
    if role == "crypto_beta_hedge":
        return "hedge ratio, equity-market hours, crypto overnight split, and execution costs"
    return "timestamp control and repeated labels"


def _next_probe(*, role: str, target: str, factor_id: str) -> str:
    if role == "residual_relative_value":
        return f"label {factor_id} as residual spread, not as broad {target} beta"
    if role == "equity_stress_control":
        return "use miner stress as a regime control before any directional crypto action"
    if role == "crypto_beta_hedge":
        return f"label {target} beta hedge with explicit equity-hours and crypto-hours windows"
    return f"keep {factor_id} as context until the factor role is clear"


def _target_for_group(group: str) -> str:
    if group == "eth_treasury_equity":
        return "ETH"
    if group in {"btc_treasury_equity", "spot_btc_etf", "btc_futures_etf", "miner"}:
        return "BTC"
    return "BTC_ETH"


def _date_gap(latest: str, current: str) -> int:
    try:
        latest_date = date.fromisoformat(latest)
        current_date = date.fromisoformat(current)
    except ValueError:
        return 0
    return (latest_date - current_date).days


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", type=Path, default=ROOT / "current_crypto_equity_proxy_context.csv")
    parser.add_argument("--ticket-path", type=Path, default=ROOT / "current_crypto_equity_proxy_paper_tickets.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_crypto_equity_factor_split.csv")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_crypto_equity_factor_split.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_crypto_equity_factor_split_rows(snapshot_path=args.snapshot_path, ticket_path=args.ticket_path)
    write_crypto_equity_factor_split_csv(rows, output_path=args.output_path)
    write_crypto_equity_factor_split_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.factor_id, row.factor_role, f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
