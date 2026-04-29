from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_SUBJECT_ID, DEFAULT_TARGET


@dataclass(frozen=True)
class SubjectEvaluationInput:
    date: str
    signal_id: str
    prediction: float
    observation: float
    evaluation_id: str | None = None
    subject_id: str = DEFAULT_SUBJECT_ID
    target_id: str = DEFAULT_TARGET
    funding_cost_bps: float | None = None
    borrow_fee_bps: float | None = None
    roll_cost_bps: float | None = None
    financing_cost_bps: float | None = None
    contract_multiplier: float | None = None
    contract_id: str | None = None
    contract_family: str | None = None
    quote_ccy: str | None = None
    collateral_ccy: str | None = None
    roll_event: dict[str, object] | None = None

EvaluationInput = SubjectEvaluationInput


def _parse_evaluation_input(item: object, *, source: Path) -> SubjectEvaluationInput:
    if not isinstance(item, dict):
        raise ValueError(f"{source}: each evaluation input must be a JSON object")

    subject_id = item.get("subject_id", item.get("asset", DEFAULT_SUBJECT_ID))
    target_id = str(item.get("target_id", DEFAULT_TARGET))

    try:
        date = str(item["date"])
        signal_id = str(item["signal_id"])
        prediction = float(item["prediction"])
        observation = float(item["observation"])
    except KeyError as exc:
        raise ValueError(f"{source}: missing required key {exc.args[0]}") from exc

    evaluation_id_obj = item.get("evaluation_id")
    evaluation_id = None if evaluation_id_obj is None else str(evaluation_id_obj)
    funding_cost_bps_obj = item.get("funding_cost_bps")
    borrow_fee_bps_obj = item.get("borrow_fee_bps")
    roll_cost_bps_obj = item.get("roll_cost_bps")
    financing_cost_bps_obj = item.get("financing_cost_bps")
    contract_multiplier_obj = item.get("contract_multiplier")
    contract_id_obj = item.get("contract_id")
    contract_family_obj = item.get("contract_family")
    quote_ccy_obj = item.get("quote_ccy")
    collateral_ccy_obj = item.get("collateral_ccy")
    roll_event_obj = item.get("roll_event")
    if roll_event_obj is not None and not isinstance(roll_event_obj, dict):
        raise ValueError(f"{source}: roll_event must be a JSON object")
    return SubjectEvaluationInput(
        date=date,
        signal_id=signal_id,
        prediction=prediction,
        observation=observation,
        evaluation_id=evaluation_id,
        subject_id=str(subject_id),
        target_id=target_id,
        funding_cost_bps=(
            None if funding_cost_bps_obj is None else float(funding_cost_bps_obj)
        ),
        borrow_fee_bps=(
            None if borrow_fee_bps_obj is None else float(borrow_fee_bps_obj)
        ),
        roll_cost_bps=(
            None if roll_cost_bps_obj is None else float(roll_cost_bps_obj)
        ),
        financing_cost_bps=(
            None
            if financing_cost_bps_obj is None
            else float(financing_cost_bps_obj)
        ),
        contract_multiplier=(
            None
            if contract_multiplier_obj is None
            else float(contract_multiplier_obj)
        ),
        contract_id=None if contract_id_obj is None else str(contract_id_obj),
        contract_family=(
            None if contract_family_obj is None else str(contract_family_obj)
        ),
        quote_ccy=None if quote_ccy_obj is None else str(quote_ccy_obj),
        collateral_ccy=(
            None if collateral_ccy_obj is None else str(collateral_ccy_obj)
        ),
        roll_event=(
            None
            if roll_event_obj is None
            else {str(key): value for key, value in roll_event_obj.items()}
        ),
    )


def load_evaluation_input(path: str | Path) -> SubjectEvaluationInput:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _parse_evaluation_input(payload, source=source)


def load_evaluation_inputs(path: str | Path) -> list[SubjectEvaluationInput]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{source}: expected a JSON array of evaluation inputs")
    return [_parse_evaluation_input(item, source=source) for item in payload]
