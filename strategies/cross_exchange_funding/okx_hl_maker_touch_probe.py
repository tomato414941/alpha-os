from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean

from strategies.cross_exchange_funding.okx_hl_book_depth import (
    BookFillCheck,
    build_book_depth_check,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MakerTouchSnapshot:
    sampled_at: str
    asset: str
    venue: str
    side: str
    best_bid: Decimal
    best_ask: Decimal
    mid_price: Decimal
    maker_quote: Decimal
    maker_edge_bps: Decimal


@dataclass(frozen=True)
class MakerTouchObservation:
    placed_at: str
    checked_at: str
    asset: str
    venue: str
    side: str
    maker_quote: Decimal
    next_best_bid: Decimal
    next_best_ask: Decimal
    touched: bool
    maker_edge_bps: Decimal


@dataclass(frozen=True)
class MakerTouchSummary:
    asset: str
    venue: str
    side: str
    observations: int
    touch_rate: float
    mean_maker_edge_bps: Decimal
    min_maker_edge_bps: Decimal
    max_maker_edge_bps: Decimal


@dataclass(frozen=True)
class MakerTouchPairObservation:
    placed_at: str
    checked_at: str
    asset: str
    okx_side: str
    hl_side: str
    okx_touched: bool
    hl_touched: bool
    both_touched: bool
    either_touched: bool
    okx_maker_edge_bps: Decimal
    hl_maker_edge_bps: Decimal


@dataclass(frozen=True)
class MakerTouchPairSummary:
    asset: str
    observations: int
    both_touch_rate: float
    either_touch_rate: float
    okx_only_touch_rate: float
    hl_only_touch_rate: float
    no_touch_rate: float
    mean_okx_maker_edge_bps: Decimal
    mean_hl_maker_edge_bps: Decimal


def collect_maker_touch_probe(
    *,
    assets: tuple[str, ...],
    samples: int,
    delay_seconds: float,
    target_notional: Decimal,
    direction_path: Path = ROOT / "okx_hl_event_window_triage.csv",
) -> tuple[MakerTouchObservation, ...]:
    directions = _read_directions(direction_path=direction_path, assets=assets)
    snapshots: list[tuple[MakerTouchSnapshot, ...]] = []
    for sample_index in range(samples):
        snapshots.append(
            tuple(
                snapshot
                for asset in assets
                for snapshot in _asset_snapshots(
                    asset=asset,
                    long_venue=directions[asset]["long_venue"],
                    target_notional=target_notional,
                )
            )
        )
        if sample_index + 1 < samples:
            time.sleep(delay_seconds)
    observations: list[MakerTouchObservation] = []
    for current, next_snapshot in zip(snapshots[:-1], snapshots[1:], strict=True):
        next_by_key = {
            (snapshot.asset, snapshot.venue): snapshot for snapshot in next_snapshot
        }
        observations.extend(
            _touch_observation(
                current=current_snapshot,
                next_snapshot=next_by_key[(current_snapshot.asset, current_snapshot.venue)],
            )
            for current_snapshot in current
        )
    return tuple(observations)


def build_pair_observations(
    observations: tuple[MakerTouchObservation, ...],
) -> tuple[MakerTouchPairObservation, ...]:
    by_window: dict[tuple[str, str, str], list[MakerTouchObservation]] = {}
    for observation in observations:
        by_window.setdefault(
            (observation.asset, observation.placed_at, observation.checked_at),
            [],
        ).append(observation)
    pair_observations = tuple(
        _pair_observation(observations=tuple(window_observations))
        for window_observations in by_window.values()
    )
    return tuple(sorted(pair_observations, key=lambda item: (item.asset, item.placed_at)))


def summarize_pair_touch(
    pair_observations: tuple[MakerTouchPairObservation, ...],
) -> tuple[MakerTouchPairSummary, ...]:
    by_asset: dict[str, list[MakerTouchPairObservation]] = {}
    for observation in pair_observations:
        by_asset.setdefault(observation.asset, []).append(observation)
    summaries = tuple(
        _summarize_pair(asset=asset, observations=tuple(asset_observations))
        for asset, asset_observations in by_asset.items()
    )
    return tuple(sorted(summaries, key=lambda item: item.both_touch_rate, reverse=True))


def summarize_maker_touch(
    observations: tuple[MakerTouchObservation, ...],
) -> tuple[MakerTouchSummary, ...]:
    by_leg: dict[tuple[str, str, str], list[MakerTouchObservation]] = {}
    for observation in observations:
        by_leg.setdefault(
            (observation.asset, observation.venue, observation.side),
            [],
        ).append(observation)
    summaries = tuple(
        _summarize_leg(
            asset=asset,
            venue=venue,
            side=side,
            observations=tuple(leg_observations),
        )
        for (asset, venue, side), leg_observations in by_leg.items()
    )
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                summary.asset,
                summary.touch_rate,
                summary.mean_maker_edge_bps,
            ),
            reverse=True,
        )
    )


def write_maker_touch_observations_csv(
    observations: tuple[MakerTouchObservation, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "placed_at",
                "checked_at",
                "asset",
                "venue",
                "side",
                "maker_quote",
                "next_best_bid",
                "next_best_ask",
                "touched",
                "maker_edge_bps",
            )
        )
        for observation in observations:
            writer.writerow(
                (
                    observation.placed_at,
                    observation.checked_at,
                    observation.asset,
                    observation.venue,
                    observation.side,
                    _fmt(observation.maker_quote),
                    _fmt(observation.next_best_bid),
                    _fmt(observation.next_best_ask),
                    observation.touched,
                    _fmt(observation.maker_edge_bps),
                )
            )
    return output_path


def write_pair_observations_csv(
    observations: tuple[MakerTouchPairObservation, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "placed_at",
                "checked_at",
                "asset",
                "okx_side",
                "hl_side",
                "okx_touched",
                "hl_touched",
                "both_touched",
                "either_touched",
                "okx_maker_edge_bps",
                "hl_maker_edge_bps",
            )
        )
        for observation in observations:
            writer.writerow(
                (
                    observation.placed_at,
                    observation.checked_at,
                    observation.asset,
                    observation.okx_side,
                    observation.hl_side,
                    observation.okx_touched,
                    observation.hl_touched,
                    observation.both_touched,
                    observation.either_touched,
                    _fmt(observation.okx_maker_edge_bps),
                    _fmt(observation.hl_maker_edge_bps),
                )
            )
    return output_path


def write_pair_summary_csv(
    summaries: tuple[MakerTouchPairSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "asset",
                "observations",
                "both_touch_rate",
                "either_touch_rate",
                "okx_only_touch_rate",
                "hl_only_touch_rate",
                "no_touch_rate",
                "mean_okx_maker_edge_bps",
                "mean_hl_maker_edge_bps",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.asset,
                    summary.observations,
                    f"{summary.both_touch_rate:.8f}",
                    f"{summary.either_touch_rate:.8f}",
                    f"{summary.okx_only_touch_rate:.8f}",
                    f"{summary.hl_only_touch_rate:.8f}",
                    f"{summary.no_touch_rate:.8f}",
                    _fmt(summary.mean_okx_maker_edge_bps),
                    _fmt(summary.mean_hl_maker_edge_bps),
                )
            )
    return output_path


def write_pair_summary_md(
    summaries: tuple[MakerTouchPairSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Maker Touch Pair Summary\n\n")
        handle.write(
            "This pairs OKX and Hyperliquid maker-touch observations by asset and "
            "sample window. Both legs must touch in the same window for a clean "
            "maker-maker entry proxy.\n\n"
        )
        handle.write(
            "| asset | obs | both touch rate | either touch rate | OKX only | HL only | no touch | mean OKX edge bps | mean HL edge bps |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for summary in summaries:
            handle.write(
                "| "
                f"{summary.asset} | "
                f"{summary.observations} | "
                f"{summary.both_touch_rate:.8f} | "
                f"{summary.either_touch_rate:.8f} | "
                f"{summary.okx_only_touch_rate:.8f} | "
                f"{summary.hl_only_touch_rate:.8f} | "
                f"{summary.no_touch_rate:.8f} | "
                f"{_fmt(summary.mean_okx_maker_edge_bps)} | "
                f"{_fmt(summary.mean_hl_maker_edge_bps)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A low both-touch rate means a maker-maker entry is unlikely to complete "
            "quickly without waiting, repricing, or crossing one leg. This still does "
            "not prove real fills because queue priority is unknown.\n"
        )
    return output_path


def write_maker_touch_summary_csv(
    summaries: tuple[MakerTouchSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "asset",
                "venue",
                "side",
                "observations",
                "touch_rate",
                "mean_maker_edge_bps",
                "min_maker_edge_bps",
                "max_maker_edge_bps",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.asset,
                    summary.venue,
                    summary.side,
                    summary.observations,
                    f"{summary.touch_rate:.8f}",
                    _fmt(summary.mean_maker_edge_bps),
                    _fmt(summary.min_maker_edge_bps),
                    _fmt(summary.max_maker_edge_bps),
                )
            )
    return output_path


def write_maker_touch_summary_md(
    summaries: tuple[MakerTouchSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Maker Touch Probe\n\n")
        handle.write(
            "This is a public-book proxy for maker feasibility. It places a virtual "
            "quote at the current best bid for buy legs or best ask for sell legs, "
            "then checks whether the next sampled opposite quote would cross it. "
            "It does not prove queue position or real fills.\n\n"
        )
        handle.write(
            "| asset | venue | side | obs | touch rate | mean maker edge bps | min edge bps | max edge bps |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for summary in summaries:
            handle.write(
                "| "
                f"{summary.asset} | "
                f"{summary.venue} | "
                f"{summary.side} | "
                f"{summary.observations} | "
                f"{summary.touch_rate:.8f} | "
                f"{_fmt(summary.mean_maker_edge_bps)} | "
                f"{_fmt(summary.min_maker_edge_bps)} | "
                f"{_fmt(summary.max_maker_edge_bps)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Low touch rates mean the candidate may require waiting, repricing, or "
            "crossing the spread. High touch rates still do not prove maker fills "
            "because queue priority and post-only behavior are unknown.\n"
        )
    return output_path


def _asset_snapshots(
    *,
    asset: str,
    long_venue: str,
    target_notional: Decimal,
) -> tuple[MakerTouchSnapshot, MakerTouchSnapshot]:
    okx_side = "buy" if long_venue == "OkxSwap" else "sell"
    hl_side = "buy" if long_venue == "HlPerp" else "sell"
    check = build_book_depth_check(
        asset=asset,
        okx_target_notional=target_notional,
        hl_target_notional=target_notional,
        okx_side=okx_side,
        hl_side=hl_side,
    )
    return (
        _snapshot_from_check(sampled_at=check.generated_at, check=check.okx_check),
        _snapshot_from_check(sampled_at=check.generated_at, check=check.hl_check),
    )


def _snapshot_from_check(
    *,
    sampled_at: str,
    check: BookFillCheck,
) -> MakerTouchSnapshot:
    maker_quote = check.best_bid if check.side == "buy" else check.best_ask
    maker_edge_bps = (
        ((check.mid_price - maker_quote) / check.mid_price * Decimal("10000"))
        if check.side == "buy"
        else ((maker_quote - check.mid_price) / check.mid_price * Decimal("10000"))
    )
    return MakerTouchSnapshot(
        sampled_at=sampled_at,
        asset=check.asset,
        venue=check.venue,
        side=check.side,
        best_bid=check.best_bid,
        best_ask=check.best_ask,
        mid_price=check.mid_price,
        maker_quote=maker_quote,
        maker_edge_bps=maker_edge_bps,
    )


def _touch_observation(
    *,
    current: MakerTouchSnapshot,
    next_snapshot: MakerTouchSnapshot,
) -> MakerTouchObservation:
    touched = (
        next_snapshot.best_ask <= current.maker_quote
        if current.side == "buy"
        else next_snapshot.best_bid >= current.maker_quote
    )
    return MakerTouchObservation(
        placed_at=current.sampled_at,
        checked_at=next_snapshot.sampled_at,
        asset=current.asset,
        venue=current.venue,
        side=current.side,
        maker_quote=current.maker_quote,
        next_best_bid=next_snapshot.best_bid,
        next_best_ask=next_snapshot.best_ask,
        touched=touched,
        maker_edge_bps=current.maker_edge_bps,
    )


def _pair_observation(
    *,
    observations: tuple[MakerTouchObservation, ...],
) -> MakerTouchPairObservation:
    by_venue = {observation.venue: observation for observation in observations}
    okx = by_venue["OkxSwap"]
    hl = by_venue["HlPerp"]
    return MakerTouchPairObservation(
        placed_at=okx.placed_at,
        checked_at=okx.checked_at,
        asset=okx.asset,
        okx_side=okx.side,
        hl_side=hl.side,
        okx_touched=okx.touched,
        hl_touched=hl.touched,
        both_touched=okx.touched and hl.touched,
        either_touched=okx.touched or hl.touched,
        okx_maker_edge_bps=okx.maker_edge_bps,
        hl_maker_edge_bps=hl.maker_edge_bps,
    )


def _summarize_leg(
    *,
    asset: str,
    venue: str,
    side: str,
    observations: tuple[MakerTouchObservation, ...],
) -> MakerTouchSummary:
    edge_values = tuple(observation.maker_edge_bps for observation in observations)
    return MakerTouchSummary(
        asset=asset,
        venue=venue,
        side=side,
        observations=len(observations),
        touch_rate=mean(1.0 if observation.touched else 0.0 for observation in observations),
        mean_maker_edge_bps=Decimal(str(mean(edge_values))),
        min_maker_edge_bps=min(edge_values),
        max_maker_edge_bps=max(edge_values),
    )


def _summarize_pair(
    *,
    asset: str,
    observations: tuple[MakerTouchPairObservation, ...],
) -> MakerTouchPairSummary:
    return MakerTouchPairSummary(
        asset=asset,
        observations=len(observations),
        both_touch_rate=mean(
            1.0 if observation.both_touched else 0.0 for observation in observations
        ),
        either_touch_rate=mean(
            1.0 if observation.either_touched else 0.0 for observation in observations
        ),
        okx_only_touch_rate=mean(
            1.0 if observation.okx_touched and not observation.hl_touched else 0.0
            for observation in observations
        ),
        hl_only_touch_rate=mean(
            1.0 if observation.hl_touched and not observation.okx_touched else 0.0
            for observation in observations
        ),
        no_touch_rate=mean(
            1.0 if not observation.either_touched else 0.0 for observation in observations
        ),
        mean_okx_maker_edge_bps=Decimal(
            str(mean(observation.okx_maker_edge_bps for observation in observations))
        ),
        mean_hl_maker_edge_bps=Decimal(
            str(mean(observation.hl_maker_edge_bps for observation in observations))
        ),
    )


def _read_directions(
    *,
    direction_path: Path,
    assets: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    asset_set = set(assets)
    with direction_path.open(newline="", encoding="utf-8") as handle:
        directions = {
            row["asset"]: row
            for row in csv.DictReader(handle)
            if row["asset"] in asset_set
        }
    missing = tuple(asset for asset in assets if asset not in directions)
    if missing:
        raise RuntimeError(f"missing directions for assets: {', '.join(missing)}")
    return directions


def _fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")).normalize(), "f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=("BTC", "ZEC"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("--target-notional", type=Decimal, default=Decimal("1000"))
    parser.add_argument(
        "--direction-path",
        type=Path,
        default=ROOT / "okx_hl_event_window_triage.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "okx_hl_maker_touch_probe.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "okx_hl_maker_touch_probe_summary.csv",
    )
    parser.add_argument(
        "--summary-md-output-path",
        type=Path,
        default=ROOT / "okx_hl_maker_touch_probe_summary.md",
    )
    parser.add_argument(
        "--pair-output-path",
        type=Path,
        default=ROOT / "okx_hl_maker_touch_pair.csv",
    )
    parser.add_argument(
        "--pair-summary-output-path",
        type=Path,
        default=ROOT / "okx_hl_maker_touch_pair_summary.csv",
    )
    parser.add_argument(
        "--pair-summary-md-output-path",
        type=Path,
        default=ROOT / "okx_hl_maker_touch_pair_summary.md",
    )
    args = parser.parse_args()

    assets = tuple(asset.upper() for asset in args.assets)
    observations = collect_maker_touch_probe(
        assets=assets,
        samples=args.samples,
        delay_seconds=args.delay_seconds,
        target_notional=args.target_notional,
        direction_path=args.direction_path,
    )
    summaries = summarize_maker_touch(observations)
    pair_observations = build_pair_observations(observations)
    pair_summaries = summarize_pair_touch(pair_observations)
    write_maker_touch_observations_csv(observations, output_path=args.output_path)
    write_maker_touch_summary_csv(summaries, output_path=args.summary_output_path)
    write_maker_touch_summary_md(summaries, output_path=args.summary_md_output_path)
    write_pair_observations_csv(pair_observations, output_path=args.pair_output_path)
    write_pair_summary_csv(pair_summaries, output_path=args.pair_summary_output_path)
    write_pair_summary_md(
        pair_summaries,
        output_path=args.pair_summary_md_output_path,
    )
    for summary in summaries:
        print(
            summary.asset,
            summary.venue,
            summary.side,
            f"touch={summary.touch_rate:.4f}",
            f"edge_bps={_fmt(summary.mean_maker_edge_bps)}",
        )
    for summary in pair_summaries:
        print(
            summary.asset,
            "pair",
            f"both_touch={summary.both_touch_rate:.4f}",
            f"either_touch={summary.either_touch_rate:.4f}",
        )


if __name__ == "__main__":
    main()
