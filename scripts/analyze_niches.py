"""Analyze alpha input signals and niche grouping."""
import re
from collections import Counter, defaultdict

from alpha_os.alpha.registry import AlphaRegistry
from alpha_os.config import DATA_DIR
from alpha_os.dsl.tokens import (
    UNARY_OPS, BINARY_OPS, ROLLING_OPS, PAIR_ROLLING_OPS, ALLOWED_WINDOWS,
)

reg = AlphaRegistry(DATA_DIR / "alpha_registry.db")
alphas = reg.list_active()

# Build full set of operator tokens
op_tokens = set()
op_tokens |= UNARY_OPS | BINARY_OPS
for op in ROLLING_OPS:
    op_tokens.add(op)
    for w in ALLOWED_WINDOWS:
        op_tokens.add(f"{op}_{w}")
for op in PAIR_ROLLING_OPS:
    op_tokens.add(op)
    for w in ALLOWED_WINDOWS:
        op_tokens.add(f"{op}_{w}")


def extract_signals(expr_str):
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr_str)
    return sorted(set(t for t in tokens if t not in op_tokens))


signal_counts = []
signal_freq = Counter()
unique_signal_sets = set()

for a in alphas:
    sigs = extract_signals(a.expression)
    signal_counts.append(len(sigs))
    for s in sigs:
        signal_freq[s] += 1
    unique_signal_sets.add(frozenset(sigs))

print(f"Active alphas: {len(alphas)}")
print()

print("=== Input signals per alpha ===")
dist = Counter(signal_counts)
for k in sorted(dist.keys()):
    pct = dist[k] * 100 / len(alphas)
    bar = "#" * int(pct / 2)
    print(f"  {k} signals: {dist[k]:>5} alphas ({pct:5.1f}%) {bar}")

print(f"\nUnique signal sets: {len(unique_signal_sets)}")
print(f"Unique signals used: {len(signal_freq)}")
print()

print("=== Top 20 most used signals ===")
for sig, count in signal_freq.most_common(20):
    print(f"  {sig}: {count} ({count * 100 / len(alphas):.1f}%)")

print()
print("=== Niche analysis (grouped by input signal set) ===")
niche_sizes = defaultdict(int)
for a in alphas:
    sigs = frozenset(extract_signals(a.expression))
    niche_sizes[sigs] += 1

top_niches = sorted(niche_sizes.items(), key=lambda x: -x[1])[:15]
print(f"Total niches: {len(niche_sizes)}")
print("Top 15 largest niches:")
for sigs, count in top_niches:
    label = ", ".join(sorted(sigs)) if sigs else "(constants only)"
    print(f"  {count:>4} alphas: {{{label}}}")

print(f"\nNiche size distribution:")
sizes = sorted(niche_sizes.values(), reverse=True)
buckets = [("1", lambda s: s == 1), ("2-5", lambda s: 2 <= s <= 5),
           ("6-20", lambda s: 6 <= s <= 20), ("21-100", lambda s: 21 <= s <= 100),
           ("100+", lambda s: s > 100)]
for label, pred in buckets:
    n = sum(1 for s in sizes if pred(s))
    print(f"  {label:>6} alphas/niche: {n} niches")

# Cumulative: how many alphas are in niches of size > N?
print(f"\nCumulative coverage:")
total = len(alphas)
covered = 0
for sigs, count in top_niches[:10]:
    covered += count
    print(f"  Top {top_niches.index((sigs, count)) + 1} niches: {covered} alphas ({covered * 100 / total:.1f}%)")
