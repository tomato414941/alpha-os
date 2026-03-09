from __future__ import annotations

import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .replay import ReplayExperimentRun, ReplayExperimentSpec, run_replay_experiment


@dataclass(frozen=True)
class ReplayMatrixSpec:
    defaults: dict[str, Any]
    experiments: list[ReplayExperimentSpec]


def _flatten_overrides(
    payload: dict[str, Any] | None,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_overrides(value, prefix=path))
        else:
            flattened[path] = value
    return flattened


def _merge_overrides(
    base: dict[str, Any],
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base)
    merged.update(_flatten_overrides(extra))
    return merged


def load_replay_matrix(path: Path) -> ReplayMatrixSpec:
    payload = tomllib.loads(path.read_text())
    defaults = dict(payload.get("defaults", {}))
    base_overrides = _flatten_overrides(defaults.get("overrides", {}))

    experiments: list[ReplayExperimentSpec] = []
    for item in payload.get("experiment", []):
        spec = ReplayExperimentSpec(
            name=item["name"],
            asset=item.get("asset", defaults.get("asset", "BTC")),
            start_date=item.get("start_date", defaults["start_date"]),
            end_date=item.get("end_date", defaults["end_date"]),
            config_path=(
                Path(item["config_path"])
                if item.get("config_path")
                else (Path(defaults["config_path"]) if defaults.get("config_path") else None)
            ),
            registry_mode=item.get("registry_mode", defaults.get("registry_mode", "current")),
            admission_source=item.get(
                "admission_source",
                defaults.get("admission_source", "candidates"),
            ),
            fail_state=item.get("fail_state", defaults.get("fail_state", "rejected")),
            deployment_mode=item.get(
                "deployment_mode",
                defaults.get("deployment_mode", defaults.get("universe_mode", "current")),
            ),
            sizing_mode=item.get("sizing_mode", defaults.get("sizing_mode", "runtime")),
            overrides=_merge_overrides(base_overrides, item.get("overrides")),
            notes=item.get("notes", defaults.get("notes", "")),
        )
        experiments.append(spec)

    return ReplayMatrixSpec(defaults=defaults, experiments=experiments)


def run_replay_matrix(
    matrix: ReplayMatrixSpec,
    *,
    max_workers: int = 1,
) -> list[ReplayExperimentRun]:
    if max_workers <= 1:
        return [run_replay_experiment(spec) for spec in matrix.experiments]

    indexed_specs = list(enumerate(matrix.experiments))
    completed: dict[int, ReplayExperimentRun] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_replay_experiment, spec): index
            for index, spec in indexed_specs
        }
        for future in as_completed(futures):
            index = futures[future]
            completed[index] = future.result()
    return [completed[index] for index, _ in indexed_specs]
