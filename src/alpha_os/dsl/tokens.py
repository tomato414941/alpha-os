from enum import Enum


class OpType(Enum):
    UNARY = "unary"
    BINARY = "binary"
    ROLLING = "rolling"
    PAIR_ROLLING = "pair_rolling"


UNARY_OPS = {"neg", "abs", "sign", "log", "zscore"}
BINARY_OPS = {"add", "sub", "mul", "div", "max", "min"}
ROLLING_OPS = {"mean", "std", "ts_max", "ts_min", "delta", "roc", "rank", "ema"}
PAIR_ROLLING_OPS = {"corr", "cov"}
ALLOWED_WINDOWS = [5, 10, 20, 30, 60]
