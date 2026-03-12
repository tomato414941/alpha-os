"""Helpers for identifying comparable runtime profiles."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from .config import PROJECT_DIR, Config


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    git_commit: str
    config_id: str
    deployed_set_id: str
    payload: dict[str, Any]

    @property
    def short_id(self) -> str:
        return self.profile_id[:12]


@lru_cache(maxsize=1)
def git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _runtime_config_payload(cfg: Config) -> dict[str, Any]:
    return {
        "trading": {
            "mode": cfg.trading.mode,
            "initial_capital": cfg.trading.initial_capital,
        },
        "paper": asdict(cfg.paper),
        "risk": asdict(cfg.risk),
        "forward": asdict(cfg.forward),
        "live_quality": asdict(cfg.live_quality),
        "deployment": asdict(cfg.deployment),
        "lifecycle": asdict(cfg.lifecycle),
        "execution": asdict(cfg.execution),
        "regime": asdict(cfg.regime),
    }


def build_runtime_profile(
    *,
    asset: str,
    config: Config,
    deployed_alpha_ids: list[str],
    extra: dict[str, Any] | None = None,
    commit: str | None = None,
) -> RuntimeProfile:
    resolved_commit = git_commit() if commit is None else commit
    deployed_ids = sorted(deployed_alpha_ids)
    config_payload = _runtime_config_payload(config)
    config_raw = json.dumps(config_payload, sort_keys=True, separators=(",", ":"))
    deployed_raw = json.dumps(deployed_ids, sort_keys=True, separators=(",", ":"))
    config_id = hashlib.sha1(config_raw.encode()).hexdigest()
    deployed_set_id = hashlib.sha1(deployed_raw.encode()).hexdigest()
    profile_payload = {
        "asset": asset.upper(),
        "config": config_payload,
        "deployed_alpha_ids": deployed_ids,
    }
    if extra:
        profile_payload["extra"] = extra
    raw = json.dumps(profile_payload, sort_keys=True, separators=(",", ":"))
    payload = {
        **profile_payload,
        "git_commit": resolved_commit,
    }
    return RuntimeProfile(
        profile_id=hashlib.sha1(raw.encode()).hexdigest(),
        git_commit=resolved_commit,
        config_id=config_id,
        deployed_set_id=deployed_set_id,
        payload=payload,
    )
