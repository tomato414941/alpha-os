from __future__ import annotations

from ..dsl import parse, to_string

_HANDCRAFTED_ALPHA_SETS: dict[str, dict[str, list[str]]] = {
    "BTC": {
        "baseline": [
            "(roc_20 btc_ohlcv)",
            "(neg (zscore (roc_5 btc_ohlcv)))",
            "(sub (roc_20 btc_ohlcv) (roc_20 sp500))",
            "(if_gt vix_close 25.0 (neg (roc_10 btc_ohlcv)) (roc_10 btc_ohlcv))",
            "(if_gt fear_greed 70.0 (neg btc_ohlcv) btc_ohlcv)",
            "(sub (zscore btc_mempool_size) (zscore btc_hashrate))",
            "(sub funding_rate_btc (mean_5 funding_rate_btc))",
            "(neg book_imbalance_btc)",
            "(neg spread_bps_btc)",
            "(if_gt vpin_btc 0.8 (neg trade_flow_btc) trade_flow_btc)",
        ],
    },
}


def list_handcrafted_sets(asset: str) -> list[str]:
    return sorted(_HANDCRAFTED_ALPHA_SETS.get(asset.upper(), {}))


def get_handcrafted_expressions(asset: str, alpha_set: str) -> list[str]:
    sets = _HANDCRAFTED_ALPHA_SETS.get(asset.upper())
    if sets is None:
        raise KeyError(f"unsupported asset for handcrafted sets: {asset}")
    expressions = sets.get(alpha_set)
    if expressions is None:
        raise KeyError(f"unknown handcrafted set for {asset}: {alpha_set}")
    return [to_string(parse(expression)) for expression in expressions]
