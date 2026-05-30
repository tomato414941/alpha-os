# Equal-weight long-only allocator adoption boundary

## Status

Closed as obsolete.

`EqualWeightLongOnlyAllocator` was introduced as a narrower alternative to the
old rich portfolio sizing path. That path, `portfolio_sizing_policy.py`, has
now been removed instead of being wired into a new route.

If equal-weight long-only behavior is needed again, implement it as a concrete
`TradingStrategy` or as an internal component of one. Do not revive the old
portfolio sizing policy layer only to host it.
