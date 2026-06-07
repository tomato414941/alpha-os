from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class CrossLaneCandidate:
    asset: str
    lanes: tuple[str, ...]
    positive_labels: tuple[str, ...]
    negative_labels: tuple[str, ...]
    pending_labels: tuple[str, ...]
    lead_score: float
    note: str


def build_cross_lane_candidates() -> tuple[CrossLaneCandidate, ...]:
    assets: dict[str, dict[str, list[str]]] = {}
    scores: dict[str, float] = {}

    _add_hl_candidate_labels(assets, scores)
    _add_okx_pressure_labels(assets, scores)
    _add_okx_liquidation_labels(assets, scores)
    _add_l2_imbalance_monitor_labels(assets, scores)
    _add_sector_rotation_labels(assets, scores)
    _add_sector_perp_context(assets, scores)
    _add_on_chain_flow_labels(assets, scores)

    candidates = tuple(
        _build_candidate(asset=asset, fields=fields, lead_score=scores.get(asset, 0.0))
        for asset, fields in assets.items()
    )
    return tuple(
        sorted(
            candidates,
            key=lambda row: (
                row.lead_score,
                len(row.lanes),
                len(row.positive_labels),
                -len(row.negative_labels),
            ),
            reverse=True,
        )
    )


def write_cross_lane_candidates(
    candidates: tuple[CrossLaneCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "lanes",
                "positive_labels",
                "negative_labels",
                "pending_labels",
                "lead_score",
                "note",
            )
        )
        for row in candidates:
            writer.writerow(
                (
                    row.asset,
                    ";".join(row.lanes),
                    ";".join(row.positive_labels),
                    ";".join(row.negative_labels),
                    ";".join(row.pending_labels),
                    f"{row.lead_score:.4f}",
                    row.note,
                )
            )
    return output_path


def write_cross_lane_candidates_md(
    candidates: tuple[CrossLaneCandidate, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Cross-Lane Candidate Review\n\n")
        handle.write(
            "This consolidates current candidate screens and first short-horizon "
            "labels. It is a triage board, not a deployable strategy ranking.\n\n"
        )
        handle.write(
            "| asset | score | lanes | positive labels | negative labels | pending labels | note |\n"
        )
        handle.write("| --- | ---: | --- | --- | --- | --- | --- |\n")
        for row in candidates[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.lead_score:.4f} | "
                f"{'; '.join(row.lanes)} | "
                f"{'; '.join(row.positive_labels)} | "
                f"{'; '.join(row.negative_labels)} | "
                f"{'; '.join(row.pending_labels)} | "
                f"{row.note} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Higher score means more current evidence survived a short label or "
            "appeared in multiple lanes. A negative label does not kill a candidate "
            "if another PnL component, such as funding, is still unmodeled.\n"
        )
    return output_path


def _add_hl_candidate_labels(
    assets: dict[str, dict[str, list[str]]],
    scores: dict[str, float],
) -> None:
    path = ROOT / "current_hl_signal_forward_label_summary.csv"
    for row in _read_rows(path):
        asset = row["asset"]
        _add_lane(assets, asset, "hl_candidate_label")
        mean15 = _float_or_none(row.get("mean_return_15m", ""))
        label_name = f"hl15={'' if mean15 is None else f'{mean15:.4f}'}"
        if mean15 is None:
            _add_pending(assets, asset, "hl15")
        elif mean15 > 0.0:
            _add_positive(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) + 2.0 + min(mean15 * 50.0, 2.0)
        else:
            _add_negative(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) - min(abs(mean15) * 25.0, 1.0)


def _add_okx_pressure_labels(
    assets: dict[str, dict[str, list[str]]],
    scores: dict[str, float],
) -> None:
    path = STRATEGIES_ROOT / "perp_market_map" / "current_okx_perp_pressure_forward_labels.csv"
    for row in _read_rows(path):
        asset = row["asset"]
        _add_lane(assets, asset, "okx_pressure")
        direction15 = _float_or_none(row.get("directional_return_15m", ""))
        score = float(row.get("pressure_score") or 0.0)
        scores[asset] = scores.get(asset, 0.0) + min(score / 500.0, 2.0)
        label_name = f"okx_pressure15={'' if direction15 is None else f'{direction15:.4f}'}"
        if direction15 is None:
            _add_pending(assets, asset, "okx_pressure15")
        elif direction15 > 0.0:
            _add_positive(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) + min(direction15 * 100.0, 1.0)
        else:
            _add_negative(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) - min(abs(direction15) * 50.0, 1.0)


def _add_okx_liquidation_labels(
    assets: dict[str, dict[str, list[str]]],
    scores: dict[str, float],
) -> None:
    path = STRATEGIES_ROOT / "liquidation_flow" / "current_okx_liquidation_forward_labels.csv"
    for row in _read_rows(path):
        asset = row["asset"]
        _add_lane(assets, asset, "okx_liquidation")
        continuation15 = _float_or_none(row.get("continuation_return_15m", ""))
        label_name = f"liq_cont15={'' if continuation15 is None else f'{continuation15:.4f}'}"
        if continuation15 is None:
            _add_pending(assets, asset, "liq_cont15")
        elif continuation15 > 0.0:
            _add_positive(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) + 1.0 + min(continuation15 * 100.0, 2.0)
        else:
            _add_negative(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) - min(abs(continuation15) * 50.0, 1.0)


def _add_l2_imbalance_monitor_labels(
    assets: dict[str, dict[str, list[str]]],
    scores: dict[str, float],
) -> None:
    monitor_path = STRATEGIES_ROOT / "market_making" / "current_l2_imbalance_monitor_summary.csv"
    labels_by_asset = _l2_forward_labels_by_asset()
    for row in _read_rows(monitor_path):
        observations = int(row.get("observations") or "0")
        persistence = float(row.get("direction_persistence_rate") or "0")
        mean_abs_imbalance = float(row.get("mean_abs_imbalance_10_bps") or "0")
        if observations < 3 or persistence < 1.0:
            continue
        asset = row["asset"]
        _add_lane(assets, asset, "l2_imbalance_monitor")
        scores[asset] = scores.get(asset, 0.0) + min(
            persistence * mean_abs_imbalance * 2.0,
            1.5,
        )
        label = labels_by_asset.get(asset)
        direction15 = None if label is None else _float_or_none(
            label.get("directional_return_15m", "")
        )
        label_name = f"l2_imbalance15={'' if direction15 is None else f'{direction15:.4f}'}"
        if direction15 is None:
            _add_pending(assets, asset, "l2_imbalance15")
        elif direction15 > 0.0:
            _add_positive(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) + min(direction15 * 100.0, 1.5)
        else:
            _add_negative(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) - min(abs(direction15) * 50.0, 1.0)


def _add_sector_rotation_labels(
    assets: dict[str, dict[str, list[str]]],
    scores: dict[str, float],
) -> None:
    path = STRATEGIES_ROOT / "sector_rotation" / "current_category_tradable_forward_labels.csv"
    for row in _read_rows(path):
        asset = row["symbol"]
        if not asset or row.get("label_status") != "tradable_labeled":
            continue
        _add_lane(assets, asset, "sector_rotation")
        direction15 = _float_or_none(row.get("directional_return_15m", ""))
        category_score = float(row.get("score") or "0")
        scores[asset] = scores.get(asset, 0.0) + min(category_score / 50.0, 1.0)
        label_name = (
            f"sector15={'' if direction15 is None else f'{direction15:.4f}'}"
            f":{row.get('category_name', '')}"
        )
        if direction15 is None:
            _add_pending(assets, asset, "sector15")
        elif direction15 > 0.0:
            _add_positive(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) + min(direction15 * 100.0, 1.0)
        else:
            _add_negative(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) - min(abs(direction15) * 50.0, 1.0)


def _add_on_chain_flow_labels(
    assets: dict[str, dict[str, list[str]]],
    scores: dict[str, float],
) -> None:
    path = STRATEGIES_ROOT / "on_chain_flow" / "current_chain_tvl_flow_forward_labels.csv"
    for row in _read_rows(path):
        asset = row["token_symbol"]
        _add_lane(assets, asset, "on_chain_flow")
        direction15 = _float_or_none(row.get("directional_return_15m", ""))
        week_change = abs(float(row.get("week_change_pct") or "0"))
        scores[asset] = scores.get(asset, 0.0) + min(week_change * 2.0, 1.0)
        label_name = (
            f"chain15={'' if direction15 is None else f'{direction15:.4f}'}"
            f":{row.get('venue', '')}:{row.get('action', '')}"
        )
        if direction15 is None:
            _add_pending(assets, asset, "chain15")
        elif direction15 > 0.0:
            _add_positive(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) + min(direction15 * 100.0, 1.0)
        else:
            _add_negative(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) - min(abs(direction15) * 50.0, 1.0)


def _add_sector_perp_context(
    assets: dict[str, dict[str, list[str]]],
    scores: dict[str, float],
) -> None:
    path = STRATEGIES_ROOT / "sector_rotation" / "current_category_perp_context.csv"
    for row in _read_rows(path):
        asset = row["symbol"]
        if not asset:
            continue
        context_score = float(row.get("context_score") or "0")
        if context_score <= 0.0:
            continue
        _add_lane(assets, asset, "sector_perp_context")
        scores[asset] = scores.get(asset, 0.0) + min(context_score / 2.0, 1.5)
        direction15 = _float_or_none(row.get("directional_return_15m", ""))
        label_name = (
            f"sector_perp15={'' if direction15 is None else f'{direction15:.4f}'}"
            f":{row.get('category_name', '')}"
        )
        if direction15 is None:
            _add_pending(assets, asset, label_name)
        elif direction15 > 0.0:
            _add_positive(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) + min(direction15 * 100.0, 1.0)
        else:
            _add_negative(assets, asset, label_name)
            scores[asset] = scores.get(asset, 0.0) - min(abs(direction15) * 50.0, 1.0)


def _l2_forward_labels_by_asset() -> dict[str, dict[str, str]]:
    path = STRATEGIES_ROOT / "market_making" / "current_l2_imbalance_forward_labels.csv"
    return {row["asset"]: row for row in _read_rows(path)}


def _build_candidate(
    *,
    asset: str,
    fields: dict[str, list[str]],
    lead_score: float,
) -> CrossLaneCandidate:
    positives = tuple(fields.get("positive_labels", ()))
    negatives = tuple(fields.get("negative_labels", ()))
    pending = tuple(fields.get("pending_labels", ()))
    return CrossLaneCandidate(
        asset=asset,
        lanes=tuple(fields.get("lanes", ())),
        positive_labels=positives,
        negative_labels=negatives,
        pending_labels=pending,
        lead_score=lead_score,
        note=_note(positives=positives, negatives=negatives, pending=pending),
    )


def _note(
    *,
    positives: tuple[str, ...],
    negatives: tuple[str, ...],
    pending: tuple[str, ...],
) -> str:
    if positives and not negatives:
        return "first labels support follow-up"
    if positives and negatives:
        return "mixed evidence; isolate which source is real"
    if negatives and not positives:
        return "current short labels are weak"
    if pending:
        return "waiting for elapsed labels"
    return "screened only"


def _add_lane(assets: dict[str, dict[str, list[str]]], asset: str, lane: str) -> None:
    lanes = assets.setdefault(asset, {}).setdefault("lanes", [])
    if lane not in lanes:
        lanes.append(lane)


def _add_positive(assets: dict[str, dict[str, list[str]]], asset: str, label: str) -> None:
    assets.setdefault(asset, {}).setdefault("positive_labels", []).append(label)


def _add_negative(assets: dict[str, dict[str, list[str]]], asset: str, label: str) -> None:
    assets.setdefault(asset, {}).setdefault("negative_labels", []).append(label)


def _add_pending(assets: dict[str, dict[str, list[str]]], asset: str, label: str) -> None:
    assets.setdefault(asset, {}).setdefault("pending_labels", []).append(label)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float_or_none(value: str) -> float | None:
    return None if value == "" else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_cross_lane_candidate_review.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_cross_lane_candidate_review.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    candidates = build_cross_lane_candidates()
    write_cross_lane_candidates(candidates, output_path=args.output_path)
    write_cross_lane_candidates_md(candidates, output_path=args.md_output_path, top=args.top)
    for row in candidates[: args.top]:
        print(
            row.asset,
            f"score={row.lead_score:.4f}",
            f"lanes={len(row.lanes)}",
            f"positive={len(row.positive_labels)}",
            f"negative={len(row.negative_labels)}",
            row.note,
        )


if __name__ == "__main__":
    main()
