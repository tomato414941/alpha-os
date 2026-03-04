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

Status: skeleton — interfaces defined, not yet integrated.
"""

from .aggregator import vote_aggregate, VoteResult
from .scorer import recency_weight

__all__ = ["vote_aggregate", "VoteResult", "recency_weight"]
