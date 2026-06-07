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
