from __future__ import annotations

import argparse
import csv
from pathlib import Path

from strategies.derivatives_positioning.current_coingecko_derivatives_positioning import (
    DerivativesPositioningRow,
    build_derivatives_positioning_rows,
    fetch_derivatives,
    write_derivatives_positioning_csv,
)


ROOT = Path(__file__).resolve().parent


def build_crowding_derivatives_coverage(
    *,
    validated_path: Path = ROOT / "current_crowding_reversion_validated_candidates.csv",
    min_open_interest: float = 1_000_000.0,
    min_volume_24h: float = 100_000.0,
) -> tuple[DerivativesPositioningRow, ...]:
    target_assets = _target_assets(validated_path)
    rows = build_derivatives_positioning_rows(
        fetch_derivatives(),
        min_open_interest=min_open_interest,
        min_volume_24h=min_volume_24h,
    )
    output = tuple(row for row in rows if _asset_key(row) in target_assets)
    return tuple(sorted(output, key=lambda row: (target_assets.index(_asset_key(row)), -row.score)))


def write_crowding_derivatives_coverage_md(
    rows: tuple[DerivativesPositioningRow, ...],
    *,
    output_path: Path,
    top: int = 80,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crowding Derivatives Coverage\n\n")
        handle.write(
            "This fetches wider CoinGecko derivatives rows for the current Hyperliquid "
            "crowding-reversion candidates. It is a coverage artifact for cross-venue "
            "checks, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | market | symbol | status | OI USD | volume 24h | OI/vol | funding | basis | score |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows[:top]:
            handle.write(
                f"| {_asset_key(row)} | "
                f"{row.market} | "
                f"{row.symbol} | "
                f"{row.status} | "
                f"{row.open_interest:.0f} | "
                f"{row.volume_24h:.0f} | "
                f"{row.oi_volume_ratio:.4f} | "
                f"{row.funding_rate:.6f} | "
                f"{row.basis:.6f} | "
                f"{row.score:.4f} |\n"
            )
    return output_path


def _target_assets(path: Path) -> tuple[str, ...]:
    rows = _read_rows(path)
    assets = []
    for row in rows:
        asset = row.get("asset", "").upper()
        if asset and asset not in assets:
            assets.append(asset)
    return tuple(assets)


def _asset_key(row: DerivativesPositioningRow) -> str:
    index_id = row.index_id.upper()
    if index_id and index_id not in {"-", "UNKNOWN"}:
        return index_id.removeprefix("K")
    symbol = row.symbol.upper()
    for suffix in ("USDTM", "USDT", "-PERP", "_PERP", "-USD", "_USD", "PERP"):
        symbol = symbol.replace(suffix, "")
    return symbol.split("/")[0].split("-")[0].split("_")[0]


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated-path", type=Path, default=ROOT / "current_crowding_reversion_validated_candidates.csv")
    parser.add_argument("--min-open-interest", type=float, default=1_000_000.0)
    parser.add_argument("--min-volume-24h", type=float, default=100_000.0)
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_crowding_derivatives_coverage.csv")
    parser.add_argument("--md-output-path", type=Path, default=ROOT / "current_crowding_derivatives_coverage.md")
    args = parser.parse_args()
    rows = build_crowding_derivatives_coverage(
        validated_path=args.validated_path,
        min_open_interest=args.min_open_interest,
        min_volume_24h=args.min_volume_24h,
    )
    write_derivatives_positioning_csv(rows, output_path=args.output_path)
    write_crowding_derivatives_coverage_md(rows, output_path=args.md_output_path)
    by_asset = sorted({ _asset_key(row) for row in rows })
    print(f"rows={len(rows)} assets={len(by_asset)}")
    print(",".join(by_asset[:40]))


if __name__ == "__main__":
    main()
