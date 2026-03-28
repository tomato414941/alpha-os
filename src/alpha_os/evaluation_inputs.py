from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_ASSET, DEFAULT_TARGET


@dataclass(frozen=True)
class EvaluationInput:
    date: str
    hypothesis_id: str
    prediction: float
    observation: float
    evaluation_id: str | None = None
    asset: str = DEFAULT_ASSET
    target: str = DEFAULT_TARGET


def _parse_evaluation_input(item: object, *, source: Path) -> EvaluationInput:
    if not isinstance(item, dict):
        raise ValueError(f"{source}: each evaluation input must be a JSON object")

    asset = str(item.get("asset", DEFAULT_ASSET))
    target = str(item.get("target", DEFAULT_TARGET))
    if asset != DEFAULT_ASSET:
        raise ValueError(f"{source}: alpha-os only supports asset={DEFAULT_ASSET}")
    if target != DEFAULT_TARGET:
        raise ValueError(f"{source}: alpha-os only supports target={DEFAULT_TARGET}")

    try:
        date = str(item["date"])
        hypothesis_id = str(item["hypothesis_id"])
        prediction = float(item["prediction"])
        observation = float(item["observation"])
    except KeyError as exc:
        raise ValueError(f"{source}: missing required key {exc.args[0]}") from exc

    evaluation_id_obj = item.get("evaluation_id")
    evaluation_id = None if evaluation_id_obj is None else str(evaluation_id_obj)
    return EvaluationInput(
        date=date,
        hypothesis_id=hypothesis_id,
        prediction=prediction,
        observation=observation,
        evaluation_id=evaluation_id,
        asset=asset,
        target=target,
    )


def load_evaluation_input(path: str | Path) -> EvaluationInput:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _parse_evaluation_input(payload, source=source)


def load_evaluation_inputs(path: str | Path) -> list[EvaluationInput]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{source}: expected a JSON array of evaluation inputs")
    return [_parse_evaluation_input(item, source=source) for item in payload]
