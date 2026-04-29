from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PriceFeaturePlane:
    daily_close: pd.Series
    daily_returns: pd.Series
    daily_volume: pd.Series | None
    dates: tuple[str, ...]
    date_to_index: dict[str, int]
    extra_observables: dict[str, pd.Series] = field(default_factory=dict)
    metadata_observables: dict[str, pd.Series] = field(default_factory=dict)
    _signal_cache: dict[tuple[str, int], pd.Series] = field(default_factory=dict)
    _observation_cache: dict[tuple[int, str], pd.Series] = field(default_factory=dict)
    _vol_cache: dict[int, pd.Series] = field(default_factory=dict)
    _volume_cache: dict[int, pd.Series] = field(default_factory=dict)

    def clone(self) -> "PriceFeaturePlane":
        return PriceFeaturePlane(
            daily_close=self.daily_close.copy(deep=True),
            daily_returns=self.daily_returns.copy(deep=True),
            daily_volume=(
                None
                if self.daily_volume is None
                else self.daily_volume.copy(deep=True)
            ),
            dates=self.dates,
            date_to_index=dict(self.date_to_index),
            extra_observables={
                key: value.copy(deep=True)
                for key, value in self.extra_observables.items()
            },
            metadata_observables={
                key: value.copy(deep=True)
                for key, value in self.metadata_observables.items()
            },
            _signal_cache={
                key: value.copy(deep=True)
                for key, value in self._signal_cache.items()
            },
            _observation_cache={
                key: value.copy(deep=True)
                for key, value in self._observation_cache.items()
            },
            _vol_cache={
                key: value.copy(deep=True)
                for key, value in self._vol_cache.items()
            },
            _volume_cache={
                key: value.copy(deep=True)
                for key, value in self._volume_cache.items()
            },
        )

    @classmethod
    def from_daily_close(
        cls,
        daily_close: pd.Series,
        *,
        daily_volume: pd.Series | None = None,
        extra_observables: dict[str, pd.Series] | None = None,
        metadata_observables: dict[str, pd.Series] | None = None,
    ) -> "PriceFeaturePlane":
        normalized = daily_close.astype(float)
        dates = tuple(str(item) for item in normalized.index)
        normalized_volume = None
        if daily_volume is not None:
            aligned = daily_volume.astype(float).reindex(normalized.index)
            normalized_volume = aligned
        normalized_extra: dict[str, pd.Series] = {}
        for observable_id, series in (extra_observables or {}).items():
            normalized_extra[str(observable_id)] = series.astype(float).reindex(
                normalized.index
            )
        normalized_metadata: dict[str, pd.Series] = {}
        for observable_id, series in (metadata_observables or {}).items():
            normalized_metadata[str(observable_id)] = series.reindex(normalized.index)
        return cls(
            daily_close=normalized,
            daily_returns=normalized.pct_change(),
            daily_volume=normalized_volume,
            dates=dates,
            date_to_index={date: idx for idx, date in enumerate(dates)},
            extra_observables=normalized_extra,
            metadata_observables=normalized_metadata,
        )

    def realized_vol_series(self, *, lookback: int) -> pd.Series:
        key = int(lookback)
        cached = self._vol_cache.get(key)
        if cached is not None:
            return cached
        vol = self.daily_returns.rolling(
            window=lookback,
            min_periods=lookback,
        ).std()
        self._vol_cache[key] = vol.astype(float)
        return self._vol_cache[key]

    def observable_series(self, *, observable_id: str) -> pd.Series:
        if observable_id == "daily_close":
            return self.daily_close
        if observable_id == "daily_return":
            return self.daily_returns
        if observable_id == "daily_volume":
            if self.daily_volume is None:
                return pd.Series(index=self.daily_close.index, dtype=float)
            return self.daily_volume
        if observable_id == "front_price":
            return self.extra_observables.get("front_price", self.daily_close)
        if observable_id == "next_price":
            return self.extra_observables.get("next_price", self.daily_close)
        if observable_id == "tradable_price":
            return self.extra_observables.get("tradable_price", self.daily_close)
        if observable_id in self.extra_observables:
            return self.extra_observables[observable_id]
        if observable_id == "term_structure_slope":
            front_price = self.observable_series(observable_id="front_price")
            next_price = self.observable_series(observable_id="next_price")
            denominator = front_price.where(front_price != 0.0)
            slope = ((next_price - front_price) / denominator).where(
                denominator.notna(),
                0.0,
            )
            return slope.astype(float)
        raise ValueError(f"unsupported observable_id: {observable_id}")

    def metadata_series(self, *, observable_id: str) -> pd.Series | None:
        return self.metadata_observables.get(observable_id)

    def dollar_volume_series(self, *, lookback: int) -> pd.Series:
        key = int(lookback)
        cached = self._volume_cache.get(key)
        if cached is not None:
            return cached
        if self.daily_volume is None:
            baseline = pd.Series(1.0, index=self.daily_close.index, dtype=float)
            self._volume_cache[key] = baseline
            return baseline
        dollar_volume = (self.daily_close * self.daily_volume).rolling(
            window=lookback,
            min_periods=lookback,
        ).mean()
        self._volume_cache[key] = dollar_volume.astype(float)
        return self._volume_cache[key]

    def signal_series(self, *, kind: str, lookback: int) -> pd.Series:
        key = (kind, int(lookback))
        cached = self._signal_cache.get(key)
        if cached is not None:
            return cached

        rolling_return_mean = self.daily_returns.rolling(
            window=lookback,
            min_periods=lookback,
        ).mean()
        if kind in {"momentum", "time_series_trend"}:
            signal = rolling_return_mean
        elif kind == "reversal":
            signal = -rolling_return_mean
        elif kind == "term_structure_carry":
            term_structure = self.observable_series(
                observable_id="term_structure_slope"
            )
            signal = term_structure.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
        elif kind == "funding_carry":
            funding_rate = self.observable_series(observable_id="funding_rate")
            signal = funding_rate.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
        elif kind == "basis_carry":
            basis = self.observable_series(observable_id="basis")
            signal = basis.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
        elif kind == "value_anchor":
            valuation_ratio = self.observable_series(observable_id="valuation_ratio")
            valuation_baseline = valuation_ratio.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
            signal = -((valuation_ratio / valuation_baseline) - 1.0).where(
                valuation_baseline != 0.0,
                0.0,
            )
        elif kind == "vol_compression_breakout":
            realized_vol = self.realized_vol_series(lookback=lookback)
            vol_baseline = realized_vol.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
            momentum_signal = rolling_return_mean
            compression = 1.0 - (realized_vol / vol_baseline)
            signal = momentum_signal * compression.where(vol_baseline != 0.0, 0.0)
        elif kind == "vol_expansion_reversal":
            realized_vol = self.realized_vol_series(lookback=lookback)
            vol_baseline = realized_vol.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
            shock = (realized_vol / vol_baseline) - 1.0
            signal = -rolling_return_mean * shock.where(vol_baseline != 0.0, 0.0)
        elif kind == "momentum_low_vol":
            realized_vol = self.realized_vol_series(lookback=lookback)
            median_vol = realized_vol.rolling(
                window=lookback,
                min_periods=lookback,
            ).median()
            low_vol_gate = (realized_vol <= median_vol).astype(float)
            signal = rolling_return_mean * low_vol_gate
        elif kind == "reversal_after_shock":
            realized_vol = self.realized_vol_series(lookback=lookback)
            shock_threshold = realized_vol.rolling(
                window=lookback,
                min_periods=lookback,
            ).quantile(0.75)
            shock_gate = (realized_vol >= shock_threshold).astype(float)
            signal = -rolling_return_mean * shock_gate
        elif kind == "trend_volume_confirmation":
            momentum_signal = rolling_return_mean
            dollar_volume = self.dollar_volume_series(lookback=lookback)
            dollar_volume_baseline = dollar_volume.rolling(
                window=lookback,
                min_periods=lookback,
            ).median()
            confirmation = (dollar_volume / dollar_volume_baseline).where(
                dollar_volume_baseline != 0.0,
                1.0,
            )
            signal = momentum_signal * confirmation
        else:
            rolling_close = self.daily_close.rolling(
                window=lookback,
                min_periods=lookback,
            )
            if kind == "average_gap":
                close_mean = rolling_close.mean()
                signal = (self.daily_close / close_mean) - 1.0
            elif kind == "range_position":
                window_min = rolling_close.min()
                window_max = rolling_close.max()
                denominator = window_max - window_min
                signal = ((self.daily_close - window_min) / denominator) * 2.0 - 1.0
                signal = signal.where(denominator != 0.0, 0.0)
            else:
                raise ValueError(f"unsupported signal kind: {kind}")

        self._signal_cache[key] = signal.astype(float)
        return self._signal_cache[key]

    def inject_signal_series(
        self,
        *,
        kind: str,
        lookback: int,
        signal: pd.Series,
    ) -> None:
        aligned = signal.astype(float).reindex(self.daily_close.index)
        self._signal_cache[(kind, int(lookback))] = aligned

    def signal_frame(self, *, kind: str, lookbacks: tuple[int, ...]) -> pd.DataFrame:
        columns = {
            int(lookback): self.signal_series(kind=kind, lookback=int(lookback))
            for lookback in lookbacks
        }
        return pd.DataFrame(columns)

    def observation_series(
        self,
        *,
        horizon_days: int,
        observable_id: str = "daily_close",
    ) -> pd.Series:
        horizon = int(horizon_days)
        cache_key = (horizon, str(observable_id))
        cached = self._observation_cache.get(cache_key)
        if cached is not None:
            return cached
        base_series = self.observable_series(observable_id=observable_id).astype(float)
        future_close = base_series.shift(-horizon)
        observation = (future_close / base_series) - 1.0
        self._observation_cache[cache_key] = observation.astype(float)
        return self._observation_cache[cache_key]
