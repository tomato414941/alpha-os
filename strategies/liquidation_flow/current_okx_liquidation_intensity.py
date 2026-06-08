from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import log10
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LiquidationIntensityRow:
    timestamp: str
    asset: str
    inst_id: str
    action: str
    total_liquidation_notional: float
    open_interest_usd: float
    liquidation_to_open_interest: float
    liquidation_to_volume: float
    forced_buy_sell_imbalance: float
    cascade_score: float
    intensity_score: float
    status: str
    next_step: str


def build_liquidation_intensity_rows(
    *,
    flow_path: Path = ROOT / "current_okx_liquidation_flow.csv",
) -> tuple[LiquidationIntensityRow, ...]:
    flow_rows = _read_rows(flow_path)
    oi_by_inst_id = _fetch_open_interest_usd(
        tuple(f"{row.get('asset', '')}-USDT-SWAP" for row in flow_rows if row.get("asset"))
    )
    rows: list[LiquidationIntensityRow] = []
    for row in flow_rows:
        asset = row.get("asset", "")
        if not asset:
            continue
        inst_id = f"{asset}-USDT-SWAP"
        oi_usd = oi_by_inst_id.get(inst_id, 0.0)
        total_liquidation = _float(row.get("total_liquidation_notional"))
        liq_to_oi = total_liquidation / oi_usd if oi_usd > 0.0 else 0.0
        imbalance = _float(row.get("forced_buy_sell_imbalance"))
        cascade_score = _float(row.get("cascade_score"))
        status = _status(liq_to_oi=liq_to_oi, imbalance=imbalance, total_liquidation=total_liquidation)
        rows.append(
            LiquidationIntensityRow(
                timestamp=row.get("timestamp", ""),
                asset=asset,
                inst_id=inst_id,
                action=row.get("action", ""),
                total_liquidation_notional=total_liquidation,
                open_interest_usd=oi_usd,
                liquidation_to_open_interest=liq_to_oi,
                liquidation_to_volume=_float(row.get("liquidation_to_volume")),
                forced_buy_sell_imbalance=imbalance,
                cascade_score=cascade_score,
                intensity_score=_intensity_score(
                    liq_to_oi=liq_to_oi,
                    total_liquidation=total_liquidation,
                    imbalance=imbalance,
                    cascade_score=cascade_score,
                ),
                status=status,
                next_step=_next_step(asset=asset, action=row.get("action", ""), status=status),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.intensity_score, reverse=True))


def write_liquidation_intensity_csv(
    rows: tuple[LiquidationIntensityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "asset",
                "inst_id",
                "action",
                "total_liquidation_notional",
                "open_interest_usd",
                "liquidation_to_open_interest",
                "liquidation_to_volume",
                "forced_buy_sell_imbalance",
                "cascade_score",
                "intensity_score",
                "status",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    row.inst_id,
                    row.action,
                    f"{row.total_liquidation_notional:.8f}",
                    f"{row.open_interest_usd:.8f}",
                    f"{row.liquidation_to_open_interest:.10f}",
                    f"{row.liquidation_to_volume:.10f}",
                    f"{row.forced_buy_sell_imbalance:.8f}",
                    f"{row.cascade_score:.8f}",
                    f"{row.intensity_score:.8f}",
                    row.status,
                    row.next_step,
                )
            )
    return output_path


def write_liquidation_intensity_md(
    rows: tuple[LiquidationIntensityRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Intensity\n\n")
        handle.write(
            "This joins recent OKX liquidation notional to OKX open interest. "
            "It is an event-intensity screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | status | liq USD | OI USD | liq/OI | liq/vol | imbalance | score | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.status} | "
                f"{row.total_liquidation_notional:.0f} | "
                f"{row.open_interest_usd:.0f} | "
                f"{row.liquidation_to_open_interest:.6f} | "
                f"{row.liquidation_to_volume:.6f} | "
                f"{row.forced_buy_sell_imbalance:.6f} | "
                f"{row.intensity_score:.4f} | "
                f"{row.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "High liquidation-to-OI rows are more likely to be true forced-flow events "
            "than rows that are merely large in dollar terms. The next test is still "
            "forward labeling with depth, fees, funding, and adverse-excursion checks.\n"
        )
    return output_path


def _fetch_open_interest_usd(inst_ids: tuple[str, ...]) -> dict[str, float]:
    output: dict[str, float] = {}
    for inst_id in tuple(dict.fromkeys(inst_ids)):
        response = requests.get(
            f"{OKX_BASE_URL}/api/v5/public/open-interest",
            params={"instType": "SWAP", "instId": inst_id},
            timeout=30,
        )
        if response.status_code >= 400:
            continue
        for item in response.json().get("data", ()):
            output[str(item.get("instId", ""))] = _float(item.get("oiUsd"))
    return output


def _status(*, liq_to_oi: float, imbalance: float, total_liquidation: float) -> str:
    if liq_to_oi >= 0.002 and abs(imbalance) >= 0.75 and total_liquidation >= 10_000.0:
        return "forced_flow_oi_shock_watch"
    if liq_to_oi >= 0.0005 and total_liquidation >= 10_000.0:
        return "liquidation_oi_pressure_watch"
    return "low_liquidation_intensity_context"


def _intensity_score(
    *,
    liq_to_oi: float,
    total_liquidation: float,
    imbalance: float,
    cascade_score: float,
) -> float:
    return (liq_to_oi * 10_000.0) + (abs(imbalance) * 2.0) + log10(max(total_liquidation, 1.0)) + cascade_score


def _next_step(*, asset: str, action: str, status: str) -> str:
    if status == "low_liquidation_intensity_context":
        return f"keep {asset} as context only unless a fresh larger forced-flow event appears"
    return f"label {asset} {action} over 5m/15m/1h with OKX depth, fees, funding, and adverse excursion"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-path", type=Path, default=ROOT / "current_okx_liquidation_flow.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_okx_liquidation_intensity.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_okx_liquidation_intensity.md")
    args = parser.parse_args()

    rows = build_liquidation_intensity_rows(flow_path=args.flow_path)
    write_liquidation_intensity_csv(rows, output_path=args.output_path)
    write_liquidation_intensity_md(rows, output_path=args.md_output_path)
    for row in rows[:10]:
        print(
            row.asset,
            row.status,
            f"liq_oi={row.liquidation_to_open_interest:.6f}",
            f"score={row.intensity_score:.4f}",
        )


if __name__ == "__main__":
    main()
