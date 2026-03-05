"""Path B: High-turnover alpha voting system.

Unlike the main alpha-os pipeline (Path A) which aims to stabilize a
top-30 portfolio of long-lived alphas, this module treats all alphas as
ephemeral voters in a statistical election. No individual alpha needs to
persist — the aggregate vote is the signal.

Design goals:
- Work well when alphas are short-lived (1-5 days)
- Use all alphas (not just top-30) as voters
- Weight by recent accuracy, not historical quality
- Produce a single directional signal with confidence
"""

from .aggregator import VoteResult, vote_aggregate
from .combiner import vote_combine
from .ensemble import EnsembleResult, compute_cell_long_pcts, ensemble_sizing
from .scorer import accuracy_from_forward, recency_weight

__all__ = [
    "EnsembleResult",
    "VoteResult",
    "accuracy_from_forward",
    "compute_cell_long_pcts",
    "ensemble_sizing",
    "recency_weight",
    "vote_aggregate",
    "vote_combine",
]
