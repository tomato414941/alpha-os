from __future__ import annotations

import json
from pathlib import Path


def load_recent_sleeve_compare_history(
    path: Path,
    *,
    limit: int = 2,
) -> dict[str, list[dict[str, object]]]:
    if limit <= 0 or not path.exists():
        return {}
    payloads: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    if not payloads:
        return {}
    history: dict[str, list[dict[str, object]]] = {}
    for payload in reversed(payloads[-limit:]):
        rows = payload.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            asset = str(row.get("asset", "")).upper()
            if not asset:
                continue
            history.setdefault(asset, []).append(row)
    return history


def load_latest_sleeve_compare_rows(path: Path) -> dict[str, dict[str, object]]:
    previous_rows: dict[str, dict[str, object]] = {}
    for asset, rows in load_recent_sleeve_compare_history(path, limit=1).items():
        if rows:
            previous_rows[asset] = rows[0]
    return previous_rows


def enrich_sleeve_compare_rows(
    rows: list[dict[str, object]],
    *,
    previous_rows: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        asset = str(row["asset"]).upper()
        previous_row = previous_rows.get(asset, {})

        current_gaps = {
            str(item) for item in row.get("serious_template_gaps", []) if str(item)
        }
        previous_gaps = {
            str(item)
            for item in previous_row.get("serious_template_gaps", [])
            if str(item)
        }
        if previous_row:
            row["serious_template_gap_delta"] = (
                f"closed:{len(previous_gaps - current_gaps)},"
                f"new:{len(current_gaps - previous_gaps)}"
            )
        else:
            row["serious_template_gap_delta"] = "first"

        row["score_budget_requested"] = int(previous_row.get("score_budget_requested", 0))
        row["score_budget_effective"] = int(previous_row.get("score_budget_effective", 0))

        previous_backed = int(previous_row.get("backed", 0))
        previous_templates = int(previous_row.get("serious_template_backed", 0))
        previous_breadth = float(previous_row.get("breadth", 0.0))
        row["backed_delta"] = int(row["backed"]) - previous_backed
        row["template_backed_delta"] = int(row["serious_template_backed"]) - previous_templates
        row["breadth_delta"] = float(row["breadth"]) - previous_breadth
        enriched.append(row)
    return enriched
