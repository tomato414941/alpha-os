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
    "ETH": {
        "baseline": [
            "(roc_20 eth_btc)",
            "(neg (zscore (roc_5 eth_btc)))",
            "(sub (roc_20 eth_btc) (roc_20 sp500))",
            "(if_gt vix_close 25.0 (neg (roc_10 eth_btc)) (roc_10 eth_btc))",
            "(if_gt fear_greed 70.0 (neg eth_btc) eth_btc)",
            "(sub (roc_20 eth_btc) (roc_20 btc_ohlcv))",
            "(sub funding_rate_eth (mean_5 funding_rate_eth))",
            "(neg (zscore oi_eth_1h))",
            "(neg (zscore liq_ratio_eth_1h))",
        ],
    },
    "SOL": {
        "baseline": [
            "(roc_20 sol_usdt)",
            "(neg (zscore (roc_5 sol_usdt)))",
            "(sub (roc_20 sol_usdt) (roc_20 sp500))",
            "(if_gt vix_close 25.0 (neg (roc_10 sol_usdt)) (roc_10 sol_usdt))",
            "(if_gt fear_greed 70.0 (neg sol_usdt) sol_usdt)",
            "(sub (roc_20 sol_usdt) (roc_20 btc_ohlcv))",
            "(sub funding_rate_sol (mean_5 funding_rate_sol))",
            "(neg (zscore oi_sol_1h))",
            "(sub (roc_10 sol_usdt) (roc_10 eth_btc))",
        ],
    },
    "NVDA": {
        "baseline": [
            "(roc_20 nvda)",
            "(neg (zscore (roc_5 nvda)))",
            "(sub (roc_20 nvda) (roc_20 sp500))",
            "(sub (roc_20 nvda) (roc_20 qqq))",
            "(if_gt vix_close 25.0 (neg (roc_10 nvda)) (roc_10 nvda))",
            "(sub (zscore nvda) (zscore nasdaq))",
            "(sub (roc_10 nvda) (roc_10 amd))",
            "(neg (sub (roc_5 tsy_yield_10y) (roc_5 tsy_yield_2y)))",
        ],
    },
    "SPY": {
        "baseline": [
            "(roc_20 spy)",
            "(neg (zscore (roc_5 spy)))",
            "(sub (roc_20 spy) (roc_20 qqq))",
            "(if_gt vix_close 25.0 (neg (roc_10 spy)) (roc_10 spy))",
            "(neg (roc_10 dxy))",
            "(sub (roc_20 spy) (roc_20 tlt))",
            "(sub (zscore spy) (zscore russell2000))",
            "(neg (sub (roc_5 tsy_yield_10y) (roc_5 tsy_yield_2y)))",
            "(if_gt fear_greed 70.0 (neg spy) spy)",
        ],
    },
    "QQQ": {
        "baseline": [
            "(roc_20 qqq)",
            "(neg (zscore (roc_5 qqq)))",
            "(sub (roc_20 qqq) (roc_20 spy))",
            "(if_gt vix_close 25.0 (neg (roc_10 qqq)) (roc_10 qqq))",
            "(neg (roc_10 dxy))",
            "(sub (roc_20 qqq) (roc_20 tlt))",
            "(sub (zscore nasdaq) (zscore sp500))",
            "(neg (sub (roc_5 tsy_yield_10y) (roc_5 tsy_yield_2y)))",
        ],
    },
    "GLD": {
        "baseline": [
            "(roc_20 gld)",
            "(neg (zscore (roc_5 gld)))",
            "(roc_20 gold)",
            "(neg (roc_10 dxy))",
            "(sub (roc_20 gld) (roc_20 spy))",
            "(sub (roc_10 gld) (roc_10 tlt))",
            "(if_gt vix_close 25.0 (roc_10 gld) (neg (roc_10 gld)))",
            "(neg (roc_20 tsy_yield_10y))",
        ],
    },
    "TLT": {
        "baseline": [
            "(roc_20 tlt)",
            "(neg (zscore (roc_5 tlt)))",
            "(neg (roc_10 tsy_yield_10y))",
            "(sub (roc_10 tsy_yield_2y) (roc_10 tsy_yield_10y))",
            "(if_gt vix_close 25.0 (roc_10 tlt) (neg (roc_10 tlt)))",
            "(neg (roc_20 sp500))",
            "(sub (roc_20 tlt) (roc_20 hyg))",
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
