from __future__ import annotations

import argparse
import json
import os
import random
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BASE_URL = "https://signal-noise.taildd87b4.ts.net"


def api_key() -> str:
    for name in ("SIGNAL_NOISE_API_KEY", "ALPHA_OS_SIGNAL_NOISE_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    secret = Path("~/.secrets/signal-noise-env").expanduser()
    if secret.exists():
        for line in secret.read_text(encoding="utf-8").splitlines():
            if line.startswith("SIGNAL_NOISE_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("SIGNAL_NOISE_API_KEY not found")


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    params: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 60,
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = None
    headers = {
        "X-API-Key": api_key(),
        "Authorization": f"Bearer {api_key()}",
        "User-Agent": "alpha-os-signal-noise-experiment/0.1",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def list_signals(base_url: str) -> list[dict[str, Any]]:
    payload = request_json(base_url, "/signals")
    if not isinstance(payload, list):
        raise TypeError(f"expected list from /signals, got {type(payload).__name__}")
    return payload


def batch_signal_data(
    base_url: str,
    names: list[str],
    *,
    since: str | None,
    resolution: str | None,
    columns: list[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    payload = request_json(
        base_url,
        "/signals/batch",
        method="POST",
        body={
            "names": names,
            "since": since,
            "resolution": resolution,
            "columns": columns,
        },
        timeout=180,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict from /signals/batch, got {type(payload).__name__}")
    return payload


def selected_signal_names(
    signals: list[dict[str, Any]],
    *,
    explicit_names: list[str],
    domains: set[str],
    categories: set[str],
    signal_types: set[str],
    active_only: bool,
    min_row_count: int | None,
    max_signals: int | None,
    sample_seed: int,
) -> list[str]:
    if explicit_names:
        return explicit_names

    eligible = set()
    for signal in signals:
        if active_only and not bool(signal.get("is_active", False)):
            continue
        if domains and str(signal.get("domain", "")) not in domains:
            continue
        if categories and str(signal.get("category", "")) not in categories:
            continue
        if signal_types and str(signal.get("signal_type", "")) not in signal_types:
            continue
        if min_row_count is not None and int(signal.get("row_count") or 0) < min_row_count:
            continue
        name = str(signal.get("name", "")).strip()
        if name:
            eligible.add(name)

    names = sorted(eligible)
    if max_signals is None or max_signals >= len(names):
        return names
    rng = random.Random(sample_seed)
    return sorted(rng.sample(names, max_signals))


def batch_signal_data_many(
    base_url: str,
    names: list[str],
    *,
    since: str | None,
    resolution: str | None,
    columns: list[str] | None,
    batch_size: int,
) -> dict[str, list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    merged: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(names), batch_size):
        chunk = names[start : start + batch_size]
        payload = batch_signal_data(
            base_url,
            chunk,
            since=since,
            resolution=resolution,
            columns=columns,
        )
        merged.update(payload)
        print(
            f"fetched signals {start + 1}-{start + len(chunk)} of {len(names)}",
            flush=True,
        )
    return merged


def write_catalog(signals: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(signals)
    frame.to_csv(output, index=False)


def write_long_frame(payload: dict[str, list[dict[str, Any]]], output: Path) -> None:
    rows = []
    for name, values in payload.items():
        for row in values:
            if not isinstance(row, dict):
                continue
            rows.append({"signal_name": name, **row})
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty and "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
        frame = frame.sort_values(["signal_name", "timestamp"])
    frame.to_csv(output, index=False)


def write_summary(
    signals: list[dict[str, Any]],
    payload: dict[str, list[dict[str, Any]]] | None,
    selected_names: list[str],
    selection_note: str,
    output: Path,
) -> None:
    domains = Counter(str(s.get("domain", "")) for s in signals)
    categories = Counter(str(s.get("category", "")) for s in signals)
    types = Counter(str(s.get("signal_type", "")) for s in signals)
    lines = [
        "# Signal Noise Stream Probe",
        "",
        f"- signal_count: {len(signals)}",
        f"- selected_signal_count: {len(selected_names)}",
        f"- selection: {selection_note}",
        "",
        "## Domains",
        "",
        "```text",
        "\n".join(f"{k}: {v}" for k, v in domains.most_common(30)),
        "```",
        "",
        "## Categories",
        "",
        "```text",
        "\n".join(f"{k}: {v}" for k, v in categories.most_common(30)),
        "```",
        "",
        "## Signal Types",
        "",
        "```text",
        "\n".join(f"{k}: {v}" for k, v in types.most_common(30)),
        "```",
    ]
    if payload is not None:
        missing_names = [name for name in selected_names if name not in payload]
        lines.extend(
            [
                "",
                "## Selected Signals",
                "",
                "```text",
                "\n".join(selected_names[:500]),
                "```",
                "",
                "## Missing Fetched Signals",
                "",
                "```text",
                "\n".join(missing_names) if missing_names else "none",
                "```",
                "",
                "## Fetched Series",
                "",
                "```text",
                "\n".join(f"{name}: {len(rows)} rows" for name, rows in sorted(payload.items())),
                "```",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("SIGNAL_NOISE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--catalog-output", type=Path, required=True)
    parser.add_argument("--data-output", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--signal-type", action="append", default=[])
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--min-row-count", type=int)
    parser.add_argument("--max-signals", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--since")
    parser.add_argument("--resolution")
    parser.add_argument("--column", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signals = list_signals(str(args.base_url))
    write_catalog(signals, args.catalog_output)
    selected_names = selected_signal_names(
        signals,
        explicit_names=list(args.name),
        domains=set(args.domain),
        categories=set(args.category),
        signal_types=set(args.signal_type),
        active_only=not bool(args.include_inactive),
        min_row_count=args.min_row_count,
        max_signals=args.max_signals,
        sample_seed=args.sample_seed,
    )
    selection_note = "explicit names" if args.name else "unordered eligible set"
    if args.max_signals is not None and not args.name:
        selection_note = f"{selection_note}; seed={args.sample_seed}; max={args.max_signals}"
    payload = None
    if selected_names:
        payload = batch_signal_data_many(
            str(args.base_url),
            selected_names,
            since=args.since,
            resolution=args.resolution,
            columns=list(args.column) if args.column else None,
            batch_size=args.batch_size,
        )
        if args.data_output is None:
            raise ValueError("--data-output is required when fetching signal data")
        write_long_frame(payload, args.data_output)
    write_summary(signals, payload, selected_names, selection_note, args.summary)
    print(f"wrote catalog: {args.catalog_output}")
    if args.data_output:
        print(f"wrote data: {args.data_output}")
    print(f"wrote summary: {args.summary}")


if __name__ == "__main__":
    main()
