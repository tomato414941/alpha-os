from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import log10
from pathlib import Path

import requests


GITHUB_REPO_URL = "https://api.github.com/repos/{repo}"
GITHUB_COMMITS_URL = "https://api.github.com/repos/{repo}/commits"
GITHUB_REPOS_BY_COIN_ID = {
    "aptos": "aptos-labs/aptos-core",
    "bitcoin": "bitcoin/bitcoin",
    "chainlink": "smartcontractkit/chainlink",
    "monero": "monero-project/monero",
    "near": "near/nearcore",
    "polygon-ecosystem-token": "maticnetwork/bor",
    "pyth-network": "pyth-network/pyth-client",
    "sui": "MystenLabs/sui",
    "the-graph": "graphprotocol/graph-node",
    "the-open-network": "ton-blockchain/ton",
    "zcash": "zcash/zcash",
}
ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class ProtocolActivitySeed:
    timestamp: str
    coin_id: str
    symbol: str
    name: str
    source_tags: tuple[str, ...]
    source_score: float


@dataclass(frozen=True)
class ProtocolActivityRow:
    timestamp: str
    coin_id: str
    symbol: str
    name: str
    source_tags: str
    github_repos: int
    stars: int
    forks: int
    pull_requests_merged: int
    commit_count_4_weeks: int
    telegram_users: int
    reddit_subscribers: int
    developer_score: float
    community_score: float
    source_score: float
    score: float
    action: str
    reason: str


def build_protocol_activity_rows(
    *,
    attention_path: Path = STRATEGIES_ROOT / "news_social" / "current_attention_snapshot.csv",
    category_path: Path = STRATEGIES_ROOT / "sector_rotation" / "current_coingecko_category_rotation.csv",
    max_coins: int = 30,
) -> tuple[ProtocolActivityRow, ...]:
    seeds = _candidate_seeds(attention_path=attention_path, category_path=category_path)
    rows: list[ProtocolActivityRow] = []
    for seed in seeds[:max_coins]:
        repo = GITHUB_REPOS_BY_COIN_ID.get(seed.coin_id)
        if not repo:
            continue
        activity = fetch_github_repo_activity(repo)
        if not activity:
            continue
        rows.append(_build_row(seed=seed, activity=activity))
    return tuple(sorted(rows, key=lambda row: row.score, reverse=True))


def fetch_github_repo_activity(repo: str) -> dict[str, object]:
    response = requests.get(
        GITHUB_REPO_URL.format(repo=repo),
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    if response.status_code in {403, 404, 429}:
        return {}
    response.raise_for_status()
    repo_payload = response.json()
    commits_4w = _github_commit_count_4w(repo)
    return {
        "repo": repo,
        "stars": repo_payload.get("stargazers_count"),
        "forks": repo_payload.get("forks_count"),
        "subscribers": repo_payload.get("subscribers_count"),
        "open_issues": repo_payload.get("open_issues_count"),
        "commits_4w": commits_4w,
    }


def write_protocol_activity_csv(
    rows: tuple[ProtocolActivityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "coin_id",
                "symbol",
                "name",
                "source_tags",
                "github_repos",
                "stars",
                "forks",
                "pull_requests_merged",
                "commit_count_4_weeks",
                "telegram_users",
                "reddit_subscribers",
                "developer_score",
                "community_score",
                "source_score",
                "score",
                "action",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.coin_id,
                    row.symbol,
                    row.name,
                    row.source_tags,
                    row.github_repos,
                    row.stars,
                    row.forks,
                    row.pull_requests_merged,
                    row.commit_count_4_weeks,
                    row.telegram_users,
                    row.reddit_subscribers,
                    f"{row.developer_score:.8f}",
                    f"{row.community_score:.8f}",
                    f"{row.source_score:.8f}",
                    f"{row.score:.8f}",
                    row.action,
                    row.reason,
                )
            )
    return output_path


def write_protocol_activity_md(
    rows: tuple[ProtocolActivityRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current CoinGecko Protocol Activity\n\n")
        handle.write(
            "This joins current attention/category candidates to CoinGecko "
            "developer and community metrics. It is a non-price context screen, "
            "not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | name | action | commits 4w | stars | telegram | source | score | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.symbol} | "
                f"{row.name} | "
                f"{row.action} | "
                f"{row.commit_count_4_weeks} | "
                f"{row.stars} | "
                f"{row.telegram_users} | "
                f"{row.source_tags} | "
                f"{row.score:.6f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "High developer or community activity is not alpha by itself. It becomes "
            "useful only when it overlaps with tradable markets, attention, funding, "
            "or event context and then survives forward labels.\n"
        )
    return output_path


def _candidate_seeds(*, attention_path: Path, category_path: Path) -> tuple[ProtocolActivitySeed, ...]:
    seeds: dict[str, ProtocolActivitySeed] = {}
    for row in _read_rows(attention_path):
        if row.get("source") != "coingecko_trending":
            continue
        coin_id = row.get("asset_id", "")
        if not coin_id:
            continue
        rank = int(row.get("rank") or "999")
        source_score = max(20 - rank, 1)
        _merge_seed(
            seeds,
            ProtocolActivitySeed(
                timestamp=row["timestamp"],
                coin_id=coin_id,
                symbol=row.get("symbol", "").upper(),
                name=row.get("name", ""),
                source_tags=(f"trending_rank_{rank}",),
                source_score=float(source_score),
            ),
        )
    for row in _read_rows(category_path)[:20]:
        timestamp = row.get("timestamp", "")
        category = row.get("name", "")
        category_score = float(row.get("score") or "0")
        source_score = min(category_score / 100.0, 10.0)
        for coin_id in (row.get("top_3_coins_id") or "").split(";"):
            if not coin_id:
                continue
            _merge_seed(
                seeds,
                ProtocolActivitySeed(
                    timestamp=timestamp,
                    coin_id=coin_id,
                    symbol="",
                    name="",
                    source_tags=(f"category_{category}",),
                    source_score=source_score,
                ),
            )
    return tuple(sorted(seeds.values(), key=lambda seed: seed.source_score, reverse=True))


def _merge_seed(seeds: dict[str, ProtocolActivitySeed], seed: ProtocolActivitySeed) -> None:
    existing = seeds.get(seed.coin_id)
    if existing is None:
        seeds[seed.coin_id] = seed
        return
    seeds[seed.coin_id] = ProtocolActivitySeed(
        timestamp=min(existing.timestamp, seed.timestamp),
        coin_id=seed.coin_id,
        symbol=existing.symbol or seed.symbol,
        name=existing.name or seed.name,
        source_tags=tuple(dict.fromkeys((*existing.source_tags, *seed.source_tags))),
        source_score=existing.source_score + seed.source_score,
    )


def _build_row(*, seed: ProtocolActivitySeed, activity: dict[str, object]) -> ProtocolActivityRow:
    stars = _int(activity.get("stars"))
    forks = _int(activity.get("forks"))
    merged = 0
    commits = _int(activity.get("commits_4w"))
    telegram = 0
    reddit = 0
    developer_score = _developer_score(
        commits=commits,
        stars=stars,
        forks=forks,
        merged=merged,
        repos=1,
    )
    community_score = _community_score(telegram=telegram, reddit=reddit)
    score = developer_score + community_score + seed.source_score
    action, reason = _classify(
        commits=commits,
        telegram=telegram,
        developer_score=developer_score,
        community_score=community_score,
        source_score=seed.source_score,
    )
    return ProtocolActivityRow(
        timestamp=seed.timestamp,
        coin_id=seed.coin_id,
        symbol=seed.symbol.upper(),
        name=seed.name or seed.coin_id,
        source_tags=";".join(seed.source_tags),
        github_repos=1,
        stars=stars,
        forks=forks,
        pull_requests_merged=merged,
        commit_count_4_weeks=commits,
        telegram_users=telegram,
        reddit_subscribers=reddit,
        developer_score=developer_score,
        community_score=community_score,
        source_score=seed.source_score,
        score=score,
        action=action,
        reason=reason,
    )


def _classify(
    *,
    commits: int,
    telegram: int,
    developer_score: float,
    community_score: float,
    source_score: float,
) -> tuple[str, str]:
    if commits >= 50 and source_score >= 5.0:
        return "developer_attention_watch", "active development overlaps current attention or sector context"
    if telegram >= 25_000 and source_score >= 5.0:
        return "community_attention_watch", "large community overlaps current attention or sector context"
    if developer_score + community_score >= 5.0:
        return "protocol_activity_context", "protocol has notable non-price activity context"
    return "low_activity_context", "activity metrics are weak for this current candidate"


def _developer_score(*, commits: int, stars: int, forks: int, merged: int, repos: int) -> float:
    return (
        min(commits / 10.0, 10.0)
        + min(log10(stars + 1.0), 5.0)
        + min(log10(forks + 1.0), 4.0)
        + min(log10(merged + 1.0), 5.0)
        + min(repos, 5)
    )


def _community_score(*, telegram: int, reddit: int) -> float:
    return min(log10(telegram + 1.0), 5.0) + min(log10(reddit + 1.0), 5.0)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _int(value: object) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _github_commit_count_4w(repo: str) -> int:
    since = (datetime.now(UTC) - timedelta(days=28)).isoformat().replace("+00:00", "Z")
    response = requests.get(
        GITHUB_COMMITS_URL.format(repo=repo),
        params={"since": since, "per_page": 1},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    if response.status_code in {403, 404, 409, 422, 429}:
        return 0
    response.raise_for_status()
    link = response.headers.get("Link", "")
    last_match = re.search(r"[?&]page=(\d+)>; rel=\"last\"", link)
    if last_match:
        return int(last_match.group(1))
    return len(response.json())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-path",
        type=Path,
        default=STRATEGIES_ROOT / "news_social" / "current_attention_snapshot.csv",
    )
    parser.add_argument(
        "--category-path",
        type=Path,
        default=STRATEGIES_ROOT / "sector_rotation" / "current_coingecko_category_rotation.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_coingecko_protocol_activity.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_coingecko_protocol_activity.md",
    )
    parser.add_argument("--max-coins", type=int, default=12)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_protocol_activity_rows(
        attention_path=args.attention_path,
        category_path=args.category_path,
        max_coins=args.max_coins,
    )
    write_protocol_activity_csv(rows, output_path=args.output_path)
    write_protocol_activity_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.action, f"score={row.score:.4f}", row.reason)


if __name__ == "__main__":
    main()
