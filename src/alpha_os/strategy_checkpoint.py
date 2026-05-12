from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyCheckpoint:
    strategy_checkpoint_id: str
    strategy_id: str
    signal_train_id: str
    signal_discovery_id: str | None
    subject_set_id: str
    target_id: str
    fold_label: str
    execution_start_date: str
    execution_end_date: str
    snapshot_set_id: str
    screening_result_id: str
    compressed_belief_id: str
    survivor_signal_ids: tuple[str, ...]
    created_at: str

    def to_document(self) -> dict[str, Any]:
        document = {
            "strategy_id": self.strategy_id,
            "signal_train_id": self.signal_train_id,
            "subject_set_id": self.subject_set_id,
            "target_id": self.target_id,
            "fold_label": self.fold_label,
            "execution_start_date": self.execution_start_date,
            "execution_end_date": self.execution_end_date,
            "snapshot_set_id": self.snapshot_set_id,
            "screening_result_id": self.screening_result_id,
            "compressed_belief_id": self.compressed_belief_id,
            "survivor_signal_ids": list(self.survivor_signal_ids),
            "created_at": self.created_at,
        }
        if self.signal_discovery_id is not None:
            document["signal_discovery_id"] = self.signal_discovery_id
        return document

    @classmethod
    def from_document(
        cls,
        *,
        strategy_checkpoint_id: str,
        document: dict[str, Any],
    ) -> "StrategyCheckpoint":
        survivor_signal_ids = document.get(
            "survivor_signal_ids",
            document.get("survivor_signal_ids", []),
        )
        if not isinstance(survivor_signal_ids, list):
            raise ValueError(
                "strategy checkpoint survivor_signal_ids must be a list"
            )
        signal_discovery_id_document = document.get("signal_discovery_id")
        signal_discovery_id = (
            None
            if signal_discovery_id_document is None
            else str(signal_discovery_id_document)
        )
        strategy_id_document = document.get("strategy_id")
        if strategy_id_document is None and signal_discovery_id is None:
            raise ValueError(
                "strategy checkpoint requires strategy_id when signal_discovery_id is absent"
            )
        signal_train_id_document = document.get("signal_train_id")
        if signal_train_id_document is None:
            if signal_discovery_id is None:
                raise ValueError(
                    "strategy checkpoint requires signal_train_id when signal_discovery_id is absent"
                )
            signal_train_id = f"signal-train:{signal_discovery_id}"
        else:
            signal_train_id = str(signal_train_id_document)
        return cls(
            strategy_checkpoint_id=strategy_checkpoint_id,
            strategy_id=str(
                strategy_id_document if strategy_id_document is not None else signal_discovery_id
            ),
            signal_train_id=signal_train_id,
            signal_discovery_id=signal_discovery_id,
            subject_set_id=str(document["subject_set_id"]),
            target_id=str(document["target_id"]),
            fold_label=str(document["fold_label"]),
            execution_start_date=str(document["execution_start_date"]),
            execution_end_date=str(document["execution_end_date"]),
            snapshot_set_id=str(document["snapshot_set_id"]),
            screening_result_id=str(document["screening_result_id"]),
            compressed_belief_id=str(document["compressed_belief_id"]),
            survivor_signal_ids=tuple(
                str(item) for item in survivor_signal_ids
            ),
            created_at=str(document["created_at"]),
        )
