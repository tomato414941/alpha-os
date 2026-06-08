from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class SeedWalletFlowActionability:
    candidate_id: str
    wallet_label: str
    source_coin: str
    execution_asset: str
    action: str
    status: str
    side: str
    score: float
    fills: int
    net_buy_notional: float
    net_closed_pnl_after_fees: float
    current_position: float
    current_position_notional: float
    mark_price: str
    reason: str
    next_step: str


def build_seed_wallet_flow_actionability(
    *,
    seed_flow_path: Path = ROOT / "current_hyperliquid_seed_wallet_flow.csv",
    hyperliquid_snapshot_path: Path = STRATEGIES_ROOT / "perp_market_map" / "current_hyperliquid_snapshot.csv",
) -> tuple[SeedWalletFlowActionability, ...]:
    marks = _hyperliquid_marks(hyperliquid_snapshot_path)
    rows = tuple(
        _build_candidate(row=row, marks=marks)
        for row in _read_rows(seed_flow_path)
    )
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def write_seed_wallet_flow_actionability_csv(
    rows: tuple[SeedWalletFlowActionability, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "wallet_label",
                "source_coin",
                "execution_asset",
                "action",
                "status",
                "side",
                "score",
                "fills",
                "net_buy_notional",
                "net_closed_pnl_after_fees",
                "current_position",
                "current_position_notional",
                "mark_price",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.wallet_label,
                    row.source_coin,
                    row.execution_asset,
                    row.action,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    row.fills,
                    f"{row.net_buy_notional:.8f}",
                    f"{row.net_closed_pnl_after_fees:.8f}",
                    f"{row.current_position:.8f}",
                    f"{row.current_position_notional:.8f}",
                    row.mark_price,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_seed_wallet_flow_actionability_md(
    rows: tuple[SeedWalletFlowActionability, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Seed Wallet Flow Actionability\n\n")
        handle.write(
            "This filters public seed-wallet flow into paper-label candidates. "
            "It is not a copy-trading rule and not a verified entity model.\n\n"
        )
        handle.write(
            "| candidate | asset | side | status | score | fills | net buy USD | net PnL | position USD | reason |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.candidate_id} | "
                f"{row.execution_asset} | "
                f"{row.side} | "
                f"{row.status} | "
                f"{row.score:.2f} | "
                f"{row.fills} | "
                f"{row.net_buy_notional:.2f} | "
                f"{row.net_closed_pnl_after_fees:.2f} | "
                f"{row.current_position_notional:.2f} | "
                f"{_escape(row.reason)} |\n"
            )
        handle.write("\n## Rule\n\n")
        handle.write(
            "Only tradable assets with positive realized seed-wallet PnL become actionability candidates. "
            "Position-follow candidates need an open position; recent-flow candidates need enough fill history "
            "and a material net buy/sell imbalance.\n"
        )
    return output_path


def _build_candidate(
    *,
    row: dict[str, str],
    marks: dict[str, str],
) -> SeedWalletFlowActionability:
    source_coin = row.get("coin", "")
    execution_asset = _execution_asset(source_coin=source_coin, marks=marks)
    action = row.get("action", "")
    fills = int(float(row.get("fills") or 0))
    net_buy = _float(row.get("net_buy_notional"))
    net_pnl = _float(row.get("net_closed_pnl_after_fees"))
    position = _float(row.get("current_position"))
    position_notional = _float(row.get("current_position_notional"))
    status = _status(
        execution_asset=execution_asset,
        action=action,
        fills=fills,
        net_buy_notional=net_buy,
        net_closed_pnl_after_fees=net_pnl,
        current_position=position,
        current_position_notional=position_notional,
    )
    side = _side(action=action, net_buy_notional=net_buy, current_position=position)
    score = _score(
        status=status,
        fills=fills,
        net_buy_notional=net_buy,
        net_closed_pnl_after_fees=net_pnl,
        current_position_notional=position_notional,
    )
    return SeedWalletFlowActionability(
        candidate_id=f"{_slug(row.get('wallet_label', ''))}_{_slug(source_coin)}_wallet_flow_actionability",
        wallet_label=row.get("wallet_label", ""),
        source_coin=source_coin,
        execution_asset=execution_asset,
        action=action,
        status=status,
        side=side,
        score=score,
        fills=fills,
        net_buy_notional=net_buy,
        net_closed_pnl_after_fees=net_pnl,
        current_position=position,
        current_position_notional=position_notional,
        mark_price=marks.get(execution_asset, ""),
        reason=_reason(status=status),
        next_step=_next_step(status=status, asset=execution_asset, side=side),
    )


def _status(
    *,
    execution_asset: str,
    action: str,
    fills: int,
    net_buy_notional: float,
    net_closed_pnl_after_fees: float,
    current_position: float,
    current_position_notional: float,
) -> str:
    if not execution_asset:
        return "wallet_flow_blocked_untradable_asset"
    if net_closed_pnl_after_fees <= 0.0 or action == "reject_negative_wallet_pnl":
        return "wallet_flow_reject_negative_seed_pnl"
    if fills >= 20 and abs(current_position) > 0.0 and abs(current_position_notional) >= 250.0:
        return "wallet_position_follow_candidate"
    if fills >= 20 and abs(net_buy_notional) >= 500.0:
        return "wallet_recent_flow_candidate"
    if fills >= 5:
        return "wallet_flow_watch"
    return "wallet_flow_deprioritize"


def _score(
    *,
    status: str,
    fills: int,
    net_buy_notional: float,
    net_closed_pnl_after_fees: float,
    current_position_notional: float,
) -> float:
    base = {
        "wallet_position_follow_candidate": 58.0,
        "wallet_recent_flow_candidate": 52.0,
        "wallet_flow_watch": 40.0,
        "wallet_flow_deprioritize": 22.0,
        "wallet_flow_blocked_untradable_asset": 12.0,
        "wallet_flow_reject_negative_seed_pnl": 0.0,
    }.get(status, 0.0)
    return (
        base
        + min(max(net_closed_pnl_after_fees, 0.0) / 20.0, 25.0)
        + min(abs(net_buy_notional) / 1_000.0, 15.0)
        + min(abs(current_position_notional) / 500.0, 12.0)
        + min(fills / 20.0, 10.0)
    )


def _side(*, action: str, net_buy_notional: float, current_position: float) -> str:
    if current_position > 0.0 or "long" in action or "buy" in action:
        return "paper_long_wallet_flow"
    if current_position < 0.0 or "short" in action or "sell" in action or net_buy_notional < 0.0:
        return "paper_short_wallet_flow"
    return "observe_wallet_flow"


def _reason(*, status: str) -> str:
    if status == "wallet_position_follow_candidate":
        return "seed wallet has positive realized PnL and an open tradable position"
    if status == "wallet_recent_flow_candidate":
        return "seed wallet has positive realized PnL and material recent buy/sell imbalance"
    if status == "wallet_flow_watch":
        return "seed wallet has positive realized PnL but the tradable flow is still small"
    if status == "wallet_flow_blocked_untradable_asset":
        return "source coin does not map to a current Hyperliquid tradable asset"
    if status == "wallet_flow_reject_negative_seed_pnl":
        return "seed wallet row has negative realized PnL after fees"
    return "wallet-flow row does not meet minimum actionability thresholds"


def _next_step(*, status: str, asset: str, side: str) -> str:
    if status in {"wallet_position_follow_candidate", "wallet_recent_flow_candidate"}:
        return f"paper-label {asset} {side} over 15m/1h/4h with funding, spread/depth, and copycat-risk controls"
    if status == "wallet_flow_watch":
        return f"keep {asset} as wallet-flow context until position/flow size or repeat labels improve"
    return "do not promote this seed-wallet row; keep only as source-quality or negative-control evidence"


def _execution_asset(*, source_coin: str, marks: dict[str, str]) -> str:
    coin = source_coin.upper()
    if coin in marks:
        return coin
    if ":" in coin:
        suffix = coin.rsplit(":", 1)[-1]
        if suffix in marks:
            return suffix
    return ""


def _hyperliquid_marks(path: Path) -> dict[str, str]:
    marks: dict[str, str] = {}
    for row in _read_rows(path):
        asset = row.get("asset", "").upper()
        mark = row.get("mark_price", "")
        if asset and mark:
            marks[asset] = mark
    return marks


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


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "na"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_seed_wallet_flow_actionability.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_seed_wallet_flow_actionability.md")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_seed_wallet_flow_actionability()
    write_seed_wallet_flow_actionability_csv(rows, output_path=args.output_path)
    write_seed_wallet_flow_actionability_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.candidate_id, row.status, row.side, f"{row.score:.4f}")


if __name__ == "__main__":
    main()
