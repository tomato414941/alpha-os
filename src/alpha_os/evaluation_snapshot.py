from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class EvaluationSnapshot:
    evaluation_id: str
    subject_id: str
    asset: str
    target_id: str
    signal_id: str
    prediction_value: float
    observation_value: float
    signed_edge: float
    absolute_error: float
    input_source: str | None
    input_range_start: str | None
    input_range_end: str | None
    funding_cost_bps: float | None
    borrow_fee_bps: float | None
    roll_cost_bps: float | None
    financing_cost_bps: float | None
    contract_multiplier: float | None
    contract_id: str | None
    contract_family: str | None
    quote_ccy: str | None
    collateral_ccy: str | None
    roll_event: dict[str, object] | None
    observation_spec_id: str | None
    observable_id: str | None
    adapter_kind: str | None
    created_at: str

    def __init__(
        self,
        *,
        evaluation_id: str,
        subject_id: str,
        asset: str,
        target_id: str,
        signal_id: str | None = None,
        prediction_value: float,
        observation_value: float,
        signed_edge: float,
        absolute_error: float,
        input_source: str | None,
        input_range_start: str | None,
        input_range_end: str | None,
        funding_cost_bps: float | None = None,
        borrow_fee_bps: float | None = None,
        roll_cost_bps: float | None = None,
        financing_cost_bps: float | None = None,
        contract_multiplier: float | None = None,
        contract_id: str | None = None,
        contract_family: str | None = None,
        quote_ccy: str | None = None,
        collateral_ccy: str | None = None,
        roll_event: dict[str, object] | None = None,
        observation_spec_id: str | None = None,
        observable_id: str | None,
        adapter_kind: str | None,
        created_at: str,
    ) -> None:
        if signal_id is None:
            raise ValueError("evaluation snapshot requires signal_id")
        object.__setattr__(self, "evaluation_id", evaluation_id)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "signal_id", str(signal_id))
        object.__setattr__(self, "prediction_value", prediction_value)
        object.__setattr__(self, "observation_value", observation_value)
        object.__setattr__(self, "signed_edge", signed_edge)
        object.__setattr__(self, "absolute_error", absolute_error)
        object.__setattr__(self, "input_source", input_source)
        object.__setattr__(self, "input_range_start", input_range_start)
        object.__setattr__(self, "input_range_end", input_range_end)
        object.__setattr__(self, "funding_cost_bps", funding_cost_bps)
        object.__setattr__(self, "borrow_fee_bps", borrow_fee_bps)
        object.__setattr__(self, "roll_cost_bps", roll_cost_bps)
        object.__setattr__(self, "financing_cost_bps", financing_cost_bps)
        object.__setattr__(self, "contract_multiplier", contract_multiplier)
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "contract_family", contract_family)
        object.__setattr__(self, "quote_ccy", quote_ccy)
        object.__setattr__(self, "collateral_ccy", collateral_ccy)
        object.__setattr__(self, "roll_event", roll_event)
        object.__setattr__(self, "observation_spec_id", observation_spec_id)
        object.__setattr__(self, "observable_id", observable_id)
        object.__setattr__(self, "adapter_kind", adapter_kind)
        object.__setattr__(self, "created_at", created_at)
