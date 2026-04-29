from __future__ import annotations

import hashlib


def build_signal_train_id(
    *,
    signal_discovery_id: str | None,
) -> str:
    normalized_signal_discovery_id = "-" if signal_discovery_id is None else signal_discovery_id
    payload = normalized_signal_discovery_id
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"signal-train:{digest}"
