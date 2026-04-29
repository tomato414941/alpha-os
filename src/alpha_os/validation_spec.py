from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_SIGNAL_NOISE_BASE_URL
from .meta_aggregation_service import DEFAULT_AGGREGATION_KINDS
from .scoring import DEFAULT_METRIC_WINDOW
from .targets import list_target_definitions


@dataclass(frozen=True)
class ValidationDateRange:
    label: str
    start_date: str
    end_date: str

    def to_document(self) -> dict[str, str]:
        return {
            "label": self.label,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }

    @classmethod
    def from_document(cls, document: dict[str, str]) -> "ValidationDateRange":
        label = document.get("label")
        start_date = document.get("start_date")
        end_date = document.get("end_date")
        if not isinstance(label, str) or not label:
            raise ValueError("validation date range is missing label")
        if not isinstance(start_date, str) or not start_date:
            raise ValueError(f"validation date range {label} is missing start_date")
        if not isinstance(end_date, str) or not end_date:
            raise ValueError(f"validation date range {label} is missing end_date")
        return cls(label=label, start_date=start_date, end_date=end_date)


@dataclass(frozen=True)
class ValidationSpec:
    signal_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    date_ranges: tuple[ValidationDateRange, ...]
    metric_windows: tuple[int, ...]
    aggregation_kinds: tuple[str, ...]
    subject_set_ids: tuple[str, ...]
    base_url: str = DEFAULT_SIGNAL_NOISE_BASE_URL

    def __post_init__(self) -> None:
        if not self.subject_set_ids:
            raise ValueError("validation spec is missing subject_set_ids")

    def to_document(self) -> dict[str, object]:
        return {
            "signal_ids": list(self.signal_ids),
            "target_ids": list(self.target_ids),
            "date_ranges": [item.to_document() for item in self.date_ranges],
            "metric_windows": list(self.metric_windows),
            "aggregation_kinds": list(self.aggregation_kinds),
            "subject_set_ids": list(self.subject_set_ids),
            "base_url": self.base_url,
        }

    @classmethod
    def from_document(cls, document: dict[str, object]) -> "ValidationSpec":
        signal_ids = document.get("signal_ids")
        target_ids = document.get("target_ids")
        date_ranges = document.get("date_ranges")
        metric_windows = document.get("metric_windows")
        aggregation_kinds = document.get("aggregation_kinds")
        subject_set_ids = document.get("subject_set_ids")
        base_url = document.get("base_url", DEFAULT_SIGNAL_NOISE_BASE_URL)
        if not isinstance(signal_ids, list) or not signal_ids:
            raise ValueError("validation spec is missing signal_ids")
        if not isinstance(target_ids, list) or not target_ids:
            raise ValueError("validation spec is missing target_ids")
        if not isinstance(date_ranges, list) or not date_ranges:
            raise ValueError("validation spec is missing date_ranges")
        if not isinstance(metric_windows, list) or not metric_windows:
            raise ValueError("validation spec is missing metric_windows")
        if not isinstance(aggregation_kinds, list) or not aggregation_kinds:
            raise ValueError("validation spec is missing aggregation_kinds")
        if not isinstance(subject_set_ids, list) or not subject_set_ids:
            raise ValueError("validation spec is missing subject_set_ids")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("validation spec is missing base_url")
        return cls(
            signal_ids=tuple(str(item) for item in signal_ids),
            target_ids=tuple(str(item) for item in target_ids),
            date_ranges=tuple(
                ValidationDateRange.from_document(item)
                for item in date_ranges
                if isinstance(item, dict)
            ),
            metric_windows=tuple(int(item) for item in metric_windows),
            aggregation_kinds=tuple(str(item) for item in aggregation_kinds),
            subject_set_ids=tuple(str(item) for item in subject_set_ids),
            base_url=base_url,
        )


def load_validation_spec(path: str | Path) -> ValidationSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("validation spec must be a JSON object")
    return ValidationSpec.from_document(raw)


def default_validation_spec(*, subject_set_ids: tuple[str, ...]) -> ValidationSpec:
    return ValidationSpec(
        signal_ids=(
            "momentum_1d",
            "momentum_3d",
            "momentum_5d",
            "reversal_1d",
            "reversal_3d",
            "reversal_5d",
            "average_gap_3d",
            "average_gap_5d",
            "range_position_5d",
        ),
        target_ids=tuple(
            definition.target_id for definition in list_target_definitions()
        ),
        date_ranges=(
            ValidationDateRange(
                label="recent_90d",
                start_date="2025-12-28",
                end_date="2026-03-27",
            ),
            ValidationDateRange(
                label="recent_180d",
                start_date="2025-09-29",
                end_date="2026-03-27",
            ),
            ValidationDateRange(
                label="year_full",
                start_date="2025-03-29",
                end_date="2026-03-27",
            ),
        ),
        metric_windows=(DEFAULT_METRIC_WINDOW, 40, 60),
        aggregation_kinds=DEFAULT_AGGREGATION_KINDS,
        subject_set_ids=subject_set_ids,
        base_url=DEFAULT_SIGNAL_NOISE_BASE_URL,
    )


def write_validation_spec(path: str | Path, spec: ValidationSpec) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(spec.to_document(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path
