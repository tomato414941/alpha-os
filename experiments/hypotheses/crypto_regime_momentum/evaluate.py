from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "experiments" / "datasets" / "ds_crypto_btc_eth_daily_2024_2025"
ASSETS = ("BTCUSDT", "ETHUSDT")
EVALUATION_START = "2024-04-01"
EVALUATION_END = "2025-12-31"
START_DATE_SENSITIVITY = ("2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01")
COST_BPS = 5.0
COST_SENSITIVITY_BPS = (0.0, 5.0, 10.0, 25.0, 50.0)
CANDIDATE_VARIANTS = {
    "candidate": {},
    "no_30d_trend_filter": {"use_30d_trend": False},
    "no_funding_filter": {"use_funding_filter": False},
    "with_volatility_scaling": {"use_volatility_scaling": True},
    "with_open_interest_scaling": {"use_open_interest_scaling": True},
}


def _load_asset(asset: str) -> pd.DataFrame:
    frame = pd.read_csv(DATASET / f"{asset}.csv", parse_dates=["timestamp"])
    frame = frame.sort_values("timestamp").set_index("timestamp")
    frame.index = frame.index.tz_convert(None)
    frame["asset"] = asset
    return frame


def _candidate_position(
    frame: pd.DataFrame,
    *,
    use_30d_trend: bool = True,
    use_funding_filter: bool = True,
    use_volatility_scaling: bool = False,
    use_open_interest_scaling: bool = False,
) -> pd.Series:
    candidate = frame["baseline_position"].copy()
    if use_30d_trend:
        candidate = candidate.where(frame["return_30d"] > 0, 0.0)
    if use_funding_filter:
        candidate = candidate.where(
            ~(
                (frame["funding_rate"] > 0)
                & (frame["funding_rate"] > frame["funding_60d_median"])
            ),
            0.0,
        )
    if use_volatility_scaling:
        candidate = candidate.where(
            ~(frame["realized_vol_20d"] > frame["realized_vol_60d_median"]),
            candidate * 0.5,
        )
    if use_open_interest_scaling:
        candidate = candidate.where(
            ~((frame["open_interest_growth_7d"] > 0) & (frame["return_7d"] < 0)),
            candidate * 0.5,
        )
    return candidate


def _asset_features(asset: str) -> pd.DataFrame:
    frame = _load_asset(asset)
    close = frame["close"]

    frame["next_return"] = close.shift(-1) / close - 1.0
    frame["return_7d"] = close / close.shift(7) - 1.0
    frame["return_30d"] = close / close.shift(30) - 1.0
    frame["realized_vol_20d"] = close.pct_change().rolling(20).std()
    frame["realized_vol_60d_median"] = frame["realized_vol_20d"].rolling(60).median()
    frame["funding_60d_median"] = frame["funding_rate"].rolling(60).median()
    frame["open_interest_growth_7d"] = frame["open_interest"] / frame["open_interest"].shift(7) - 1.0

    frame["baseline_position"] = (frame["return_7d"] > 0).astype(float)
    for variant, options in CANDIDATE_VARIANTS.items():
        frame[f"{variant}_position"] = _candidate_position(frame, **options)
    return frame


def _portfolio_returns(
    frames: dict[str, pd.DataFrame],
    position_column: str,
    *,
    cost_bps: float = COST_BPS,
    start: str = EVALUATION_START,
    end: str = EVALUATION_END,
) -> pd.DataFrame:
    parts = []
    for asset, frame in frames.items():
        part = frame[["next_return", position_column]].copy()
        part.columns = [f"{asset}_next_return", f"{asset}_position"]
        parts.append(part)
    joined = pd.concat(parts, axis=1).dropna()

    position_columns = [f"{asset}_position" for asset in ASSETS]
    active_count = (joined[position_columns] > 0).sum(axis=1).replace(0, pd.NA)

    gross_return = pd.Series(0.0, index=joined.index)
    turnover = pd.Series(0.0, index=joined.index)
    previous_weights = pd.Series(0.0, index=ASSETS)
    for date, row in joined.iterrows():
        weights = pd.Series(
            {
                asset: (
                    row[f"{asset}_position"] / active_count.loc[date]
                    if pd.notna(active_count.loc[date])
                    else 0.0
                )
                for asset in ASSETS
            }
        )
        gross_return.loc[date] = sum(
            weights[asset] * row[f"{asset}_next_return"] for asset in ASSETS
        )
        turnover.loc[date] = (weights - previous_weights).abs().sum()
        previous_weights = weights

    cost = turnover * cost_bps / 10_000
    result = pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover,
            "net_return": gross_return - cost,
            "active_assets": (joined[position_columns] > 0).sum(axis=1),
        }
    )
    return result.loc[start:end]


def _single_asset_returns(
    frame: pd.DataFrame,
    position_column: str,
    *,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    returns = frame[["next_return", position_column]].dropna().copy()
    returns["turnover"] = returns[position_column].diff().abs().fillna(returns[position_column])
    returns["gross_return"] = returns[position_column] * returns["next_return"]
    returns["net_return"] = returns["gross_return"] - returns["turnover"] * cost_bps / 10_000
    return returns.loc[EVALUATION_START:EVALUATION_END]


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _summarize(name: str, returns: pd.DataFrame) -> dict[str, float | str]:
    net = returns["net_return"]
    return {
        "strategy": name,
        "days": float(len(returns)),
        "total_net_return": float((1.0 + net).prod() - 1.0),
        "mean_daily_net_return": float(net.mean()),
        "max_drawdown": _max_drawdown(net),
        "mean_daily_turnover": float(returns["turnover"].mean()),
    }


def _summarize_by_year(name: str, returns: pd.DataFrame) -> list[dict[str, float | str]]:
    rows = []
    for year, year_returns in returns.groupby(returns.index.year):
        row = _summarize(name, year_returns)
        row["period"] = str(year)
        rows.append(row)
    return rows


def _cost_sensitivity(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cost_bps in COST_SENSITIVITY_BPS:
        baseline = _portfolio_returns(frames, "baseline_position", cost_bps=cost_bps)
        candidate = _portfolio_returns(frames, "candidate_position", cost_bps=cost_bps)
        baseline_mean = baseline["net_return"].mean()
        candidate_mean = candidate["net_return"].mean()
        rows.append(
            {
                "cost_bps": cost_bps,
                "baseline_total_net_return": float((1.0 + baseline["net_return"]).prod() - 1.0),
                "candidate_total_net_return": float(
                    (1.0 + candidate["net_return"]).prod() - 1.0
                ),
                "candidate_mean_daily_edge": float(candidate_mean - baseline_mean),
            }
        )
    return pd.DataFrame(rows)


def _start_date_sensitivity(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for start in START_DATE_SENSITIVITY:
        baseline = _portfolio_returns(frames, "baseline_position", start=start)
        candidate = _portfolio_returns(frames, "candidate_position", start=start)
        baseline_mean = baseline["net_return"].mean()
        candidate_mean = candidate["net_return"].mean()
        rows.append(
            {
                "start": start,
                "end": EVALUATION_END,
                "baseline_total_net_return": float((1.0 + baseline["net_return"]).prod() - 1.0),
                "candidate_total_net_return": float(
                    (1.0 + candidate["net_return"]).prod() - 1.0
                ),
                "candidate_mean_daily_edge": float(candidate_mean - baseline_mean),
            }
        )
    return pd.DataFrame(rows)


def _asset_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for asset, frame in frames.items():
        baseline = _single_asset_returns(frame, "baseline_position")
        candidate = _single_asset_returns(frame, "candidate_position")
        baseline_mean = baseline["net_return"].mean()
        candidate_mean = candidate["net_return"].mean()
        row = _summarize(asset, candidate)
        row["baseline_total_net_return"] = float((1.0 + baseline["net_return"]).prod() - 1.0)
        row["candidate_mean_daily_edge"] = float(candidate_mean - baseline_mean)
        rows.append(row)
    return pd.DataFrame(rows)


def _ablation_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline = _portfolio_returns(frames, "baseline_position")
    baseline_mean = baseline["net_return"].mean()
    rows = []
    for variant in CANDIDATE_VARIANTS:
        returns = _portfolio_returns(frames, f"{variant}_position")
        row = _summarize(variant, returns)
        row["mean_daily_edge_vs_baseline"] = float(returns["net_return"].mean() - baseline_mean)
        rows.append(row)
    return pd.DataFrame(rows)


def _exposure_summary(name: str, returns: pd.DataFrame) -> dict[str, float | str]:
    active_assets = returns["active_assets"]
    return {
        "strategy": name,
        "days": float(len(returns)),
        "mean_active_assets": float(active_assets.mean()),
        "flat_days": float((active_assets == 0).sum()),
        "one_asset_days": float((active_assets == 1).sum()),
        "two_asset_days": float((active_assets == 2).sum()),
    }


def _scalar(frame: pd.DataFrame, *, strategy: str, column: str) -> float:
    return float(frame.loc[frame["strategy"] == strategy, column].iloc[0])


def main() -> int:
    frames = {asset: _asset_features(asset) for asset in ASSETS}
    baseline = _portfolio_returns(frames, "baseline_position")
    candidate = _portfolio_returns(frames, "candidate_position")

    summary = pd.DataFrame(
        [
            _summarize("baseline", baseline),
            _summarize("candidate", candidate),
        ]
    )
    yearly_summary = pd.DataFrame(
        [
            *_summarize_by_year("baseline", baseline),
            *_summarize_by_year("candidate", candidate),
        ]
    )
    cost_sensitivity = _cost_sensitivity(frames)
    start_date_sensitivity = _start_date_sensitivity(frames)
    asset_summary = _asset_summary(frames)
    ablation_summary = _ablation_summary(frames)
    exposure_summary = pd.DataFrame(
        [
            _exposure_summary("baseline", baseline),
            _exposure_summary("candidate", candidate),
        ]
    )
    edge = summary.loc[summary["strategy"] == "candidate", "mean_daily_net_return"].iloc[0]
    edge -= summary.loc[summary["strategy"] == "baseline", "mean_daily_net_return"].iloc[0]
    cost_50 = cost_sensitivity.loc[cost_sensitivity["cost_bps"] == 50.0].iloc[0]

    print("Crypto regime momentum first-pass comparison")
    print(f"dataset={DATASET.relative_to(ROOT)}")
    print(f"evaluation={EVALUATION_START}..{EVALUATION_END}")
    print(f"cost_bps_per_unit_turnover={COST_BPS:g}")
    print()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Yearly summary")
    print(yearly_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Cost sensitivity")
    print(cost_sensitivity.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Start date sensitivity")
    print(start_date_sensitivity.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Candidate by asset")
    print(asset_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Ablation summary")
    print(ablation_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Exposure summary")
    print(exposure_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    facts = {
        "evaluation_start": EVALUATION_START,
        "evaluation_end": EVALUATION_END,
        "cost_bps_per_unit_turnover": COST_BPS,
        "baseline_total_net_return": _scalar(
            summary,
            strategy="baseline",
            column="total_net_return",
        ),
        "candidate_total_net_return": _scalar(
            summary,
            strategy="candidate",
            column="total_net_return",
        ),
        "candidate_mean_daily_net_return_edge": float(edge),
        "candidate_max_drawdown": _scalar(
            summary,
            strategy="candidate",
            column="max_drawdown",
        ),
        "candidate_mean_daily_turnover": _scalar(
            summary,
            strategy="candidate",
            column="mean_daily_turnover",
        ),
        "candidate_total_net_return_cost_50bps": float(
            cost_50["candidate_total_net_return"]
        ),
        "candidate_flat_days": _scalar(
            exposure_summary,
            strategy="candidate",
            column="flat_days",
        ),
        "candidate_one_asset_days": _scalar(
            exposure_summary,
            strategy="candidate",
            column="one_asset_days",
        ),
        "candidate_two_asset_days": _scalar(
            exposure_summary,
            strategy="candidate",
            column="two_asset_days",
        ),
    }
    print("FACTS")
    for key, value in facts.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
