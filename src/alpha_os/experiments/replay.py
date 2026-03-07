"""Named replay experiments with persisted artifacts."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..alpha.admission_replay import (
    apply_registry_snapshot,
    load_source_records,
    materialize_admission_snapshot,
)
from ..alpha.registry import AlphaRegistry, AlphaState
from ..config import PROJECT_DIR, Config, asset_data_dir
from ..paper.simulator import run_replay


@dataclass(frozen=True)
class ReplayExperimentSpec:
    name: str
    asset: str
    start_date: str
    end_date: str
    config_path: Path | None = None
    registry_mode: str = "current"
    admission_source: str = "candidates"
    fail_state: str = AlphaState.REJECTED
    sizing_mode: str = "runtime"
    overrides: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class ReplayExperimentRun:
    experiment_id: str
    detail_path: Path
    index_path: Path
    payload: dict[str, Any]


def parse_override_assignment(raw: str) -> tuple[str, Any]:
    """Parse PATH=VALUE using TOML semantics for VALUE."""
    if "=" not in raw:
        raise ValueError(f"Invalid override {raw!r}: expected PATH=VALUE")
    key, raw_value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override {raw!r}: empty path")
    try:
        value = tomllib.loads(f"value = {raw_value.strip()}")["value"]
    except tomllib.TOMLDecodeError:
        value = raw_value.strip()
    return key, value


def apply_config_overrides(cfg: Config, overrides: dict[str, Any]) -> None:
    """Apply dotted-path overrides to a loaded Config object."""
    for path, value in overrides.items():
        target: Any = cfg
        parts = path.split(".")
        for name in parts[:-1]:
            if not hasattr(target, name):
                raise KeyError(f"Unknown config path: {path}")
            target = getattr(target, name)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"Unknown config path: {path}")
        setattr(target, leaf, value)


def experiments_dir(asset: str) -> Path:
    path = asset_data_dir(asset) / "experiments"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "experiment"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _registry_counts(db_path: Path) -> dict[str, int]:
    registry = AlphaRegistry(db_path)
    try:
        return {
            state: registry.count(state)
            for state in (
                AlphaState.CANDIDATE,
                AlphaState.ACTIVE,
                AlphaState.DORMANT,
                AlphaState.REJECTED,
            )
        }
    finally:
        registry.close()


def _serialize_result(result) -> dict[str, Any]:
    return {
        "n_days": result.n_days,
        "initial_capital": result.initial_capital,
        "final_value": result.final_value,
        "total_return": result.total_return,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "best_day": list(result.best_day),
        "worst_day": list(result.worst_day),
    }


def run_replay_experiment(spec: ReplayExperimentSpec) -> ReplayExperimentRun:
    cfg = Config.load(spec.config_path)
    apply_config_overrides(cfg, spec.overrides)

    asset = spec.asset.upper()
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{timestamp}-{_slugify(spec.name)}"
    out_dir = experiments_dir(asset)
    detail_path = out_dir / f"{experiment_id}.json"
    index_path = out_dir / "index.jsonl"

    registry_db = asset_data_dir(asset) / "alpha_registry.db"
    registry_info: dict[str, Any] = {
        "mode": spec.registry_mode,
        "source": None,
        "fail_state": None,
        "counts": _registry_counts(registry_db),
    }

    t0 = time.perf_counter()
    if spec.registry_mode == "admission":
        source_records = load_source_records(registry_db, spec.admission_source)
        snapshot, counts = materialize_admission_snapshot(
            source_records,
            cfg.to_lifecycle_config(),
            fail_state=spec.fail_state,
        )
        registry_info = {
            "mode": spec.registry_mode,
            "source": spec.admission_source,
            "fail_state": AlphaState.canonical(spec.fail_state),
            "source_rows": len(source_records),
            "counts": counts,
        }
        with tempfile.TemporaryDirectory(prefix="alpha_os_experiment_") as tmp:
            replay_db = Path(tmp) / "registry.db"
            apply_registry_snapshot(replay_db, snapshot)
            result = run_replay(
                asset=asset,
                config=cfg,
                start_date=spec.start_date,
                end_date=spec.end_date,
                registry_db=replay_db,
                sizing_mode=spec.sizing_mode,
            )
    elif spec.registry_mode == "current":
        result = run_replay(
            asset=asset,
            config=cfg,
            start_date=spec.start_date,
            end_date=spec.end_date,
            sizing_mode=spec.sizing_mode,
        )
    else:
        raise ValueError(f"Unsupported registry mode: {spec.registry_mode}")
    elapsed = time.perf_counter() - t0

    payload = {
        "experiment_id": experiment_id,
        "name": spec.name,
        "created_at": now.isoformat(),
        "git_commit": _git_commit(),
        "spec": {
            "asset": asset,
            "start_date": spec.start_date,
            "end_date": spec.end_date,
            "config_path": str(spec.config_path) if spec.config_path else None,
            "registry_mode": spec.registry_mode,
            "admission_source": spec.admission_source,
            "fail_state": spec.fail_state,
            "sizing_mode": spec.sizing_mode,
            "notes": spec.notes,
        },
        "overrides": spec.overrides,
        "resolved_config": asdict(cfg),
        "registry": registry_info,
        "result": _serialize_result(result),
        "elapsed_seconds": elapsed,
    }

    detail_path.write_text(json.dumps(payload, indent=2))
    summary = {
        "experiment_id": experiment_id,
        "name": spec.name,
        "created_at": payload["created_at"],
        "git_commit": payload["git_commit"],
        "asset": asset,
        "start_date": spec.start_date,
        "end_date": spec.end_date,
        "registry_mode": spec.registry_mode,
        "admission_source": spec.admission_source,
        "overrides": spec.overrides,
        "final_value": payload["result"]["final_value"],
        "total_return": payload["result"]["total_return"],
        "sharpe": payload["result"]["sharpe"],
        "max_drawdown": payload["result"]["max_drawdown"],
        "total_trades": payload["result"]["total_trades"],
        "detail_path": str(detail_path),
    }
    with index_path.open("a") as fh:
        fh.write(json.dumps(summary) + "\n")

    return ReplayExperimentRun(
        experiment_id=experiment_id,
        detail_path=detail_path,
        index_path=index_path,
        payload=payload,
    )
