"""Alpaca executor for US stocks and ETFs with managed position tracking."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests

from .costs import ExecutionCostModel
from .executor import Executor, Order, Fill

logger = logging.getLogger(__name__)

_SECRETS_FILE = "alpaca"

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


def _load_secrets(name: str = _SECRETS_FILE) -> dict[str, str]:
    """Load key-value pairs from ~/.secrets/{name}."""
    secret_path = Path.home() / ".secrets" / name
    if not secret_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in secret_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, val = line.split("=", 1)
            result[key.strip()] = val.strip().strip("'\"")
    return result


def _get_credentials(paper: bool = True) -> tuple[str, str]:
    """Return (api_key, api_secret) from env vars or ~/.secrets/alpaca{_real}."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if api_key and api_secret:
        return api_key, api_secret
    secrets_name = _SECRETS_FILE if paper else f"{_SECRETS_FILE}_real"
    secrets = _load_secrets(secrets_name)
    api_key = api_key or secrets.get("ALPACA_API_KEY", "")
    api_secret = api_secret or secrets.get("ALPACA_SECRET_KEY", "")
    return api_key, api_secret


def _create_session(paper: bool = True) -> tuple[requests.Session, str]:
    """Create authenticated Alpaca API session."""
    api_key, api_secret = _get_credentials(paper=paper)
    if not api_key or not api_secret:
        secrets_file = "alpaca" if paper else "alpaca_real"
        raise ValueError(
            "Alpaca API credentials not found. "
            "Set ALPACA_API_KEY/ALPACA_SECRET_KEY env vars "
            f"or create ~/.secrets/{secrets_file}"
        )
    session = requests.Session()
    session.headers.update({
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    })
    base_url = PAPER_URL if paper else LIVE_URL
    logger.info("Created Alpaca session (paper=%s)", paper)
    return session, base_url


class AlpacaExecutor(Executor):
    """Alpaca API executor for US stocks and ETFs with managed position tracking.

    Tracks only positions created by alpha-os trades, separate from the
    full Alpaca account balance. Exchange is used for order execution
    and pre-trade safety checks.
    """

    def __init__(
        self,
        paper: bool = True,
        initial_capital: float = 10000.0,
        cost_model: ExecutionCostModel | None = None,
        max_slippage_bps: float = 20.0,
    ) -> None:
        self._session, self._base_url = _create_session(paper=paper)
        self._paper = paper
        self._cost_model = cost_model or ExecutionCostModel(
            commission_pct=0.0, modeled_slippage_pct=0.05,
        )
        self._max_slippage_bps = max_slippage_bps
        self._managed_cash: float = initial_capital
        self._managed_positions: dict[str, float] = {}
        self._initial_capital = initial_capital
        self._fills: list[Fill] = []
        self._reconciliation_cash_offset: float = 0.0
        self._reconciliation_position_offsets: dict[str, float] = {}

    def _api_get(self, path: str) -> dict | list | None:
        try:
            resp = self._session.get(f"{self._base_url}{path}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Alpaca GET %s failed: %s", path, e)
            return None

    def _api_post(self, path: str, payload: dict) -> dict | None:
        try:
            resp = self._session.post(f"{self._base_url}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Alpaca POST %s failed: %s", path, e)
            return None

    def submit_order(self, order: Order) -> Fill | None:
        if order.side == "sell" and not self.supports_short:
            current_qty = self.get_position(order.symbol)
            if order.qty > current_qty + 1e-12:
                logger.warning(
                    "Cannot sell %.6f %s with only %.6f managed in long-only mode",
                    order.qty, order.symbol, current_qty,
                )
                return None

        t0 = time.perf_counter()
        data = self._api_post("/v2/orders", {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side,
            "type": order.order_type,
            "time_in_force": "day",
        })
        latency_ms = (time.perf_counter() - t0) * 1000

        if data is None:
            return None

        filled_price = float(data.get("filled_avg_price") or 0)
        filled_qty = float(data.get("filled_qty") or order.qty)
        order_id = str(data.get("id", ""))

        if filled_price <= 0:
            logger.warning("Order %s filled_avg_price is 0, using order qty", order_id)
            return None

        notional = filled_qty * filled_price
        costs = self._cost_model.estimate_order_cost(notional, include_modeled_slippage=False)

        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            qty=filled_qty,
            price=filled_price,
            order_id=order_id,
            slippage_bps=0.0,
            latency_ms=latency_ms,
            costs=costs,
        )
        self._update_managed_state(fill)
        self._fills.append(fill)
        return fill

    def _update_managed_state(self, fill: Fill) -> None:
        cost = fill.qty * fill.price
        if fill.side == "buy":
            self._managed_cash -= cost + fill.execution_cost
            self._managed_positions[fill.symbol] = (
                self._managed_positions.get(fill.symbol, 0.0) + fill.qty
            )
        elif fill.side == "sell":
            self._managed_cash += cost - fill.execution_cost
            self._managed_positions[fill.symbol] = (
                self._managed_positions.get(fill.symbol, 0.0) - fill.qty
            )

    def get_position(self, symbol: str) -> float:
        return self._managed_positions.get(symbol, 0.0)

    def get_cash(self) -> float:
        return self._managed_cash

    def get_exchange_position(self, symbol: str) -> float:
        data = self._api_get(f"/v2/positions/{symbol}")
        if data is None or not isinstance(data, dict):
            return 0.0
        return float(data.get("qty", 0))

    def get_exchange_cash(self) -> float:
        data = self._api_get("/v2/account")
        if data is None or not isinstance(data, dict):
            return 0.0
        return float(data.get("cash", 0))

    def sync_reconciliation_baseline(self, symbols: list[str]) -> None:
        self._reconciliation_cash_offset = self.get_exchange_cash() - self._managed_cash
        offsets: dict[str, float] = {}
        for symbol in symbols:
            offsets[symbol] = (
                self.get_exchange_position(symbol)
                - self._managed_positions.get(symbol, 0.0)
            )
        self._reconciliation_position_offsets = offsets

    def get_reconciled_position(self, symbol: str) -> float:
        return (
            self.get_exchange_position(symbol)
            - self._reconciliation_position_offsets.get(symbol, 0.0)
        )

    def get_reconciled_cash(self) -> float:
        return self.get_exchange_cash() - self._reconciliation_cash_offset

    def fetch_ticker_price(self, symbol: str) -> float | None:
        """Fetch last trade price from Alpaca data API."""
        try:
            resp = self._session.get(
                f"{DATA_URL}/v2/stocks/{symbol}/trades/latest",
                headers={"APCA-API-KEY-ID": self._session.headers.get("APCA-API-KEY-ID", ""),
                         "APCA-API-SECRET-KEY": self._session.headers.get("APCA-API-SECRET-KEY", "")},
            )
            resp.raise_for_status()
            data = resp.json()
            trade = data.get("trade", data)
            return float(trade.get("p", trade.get("price", 0)))
        except Exception as e:
            logger.warning("Failed to fetch ticker for %s: %s", symbol, e)
            return None

    @property
    def supports_short(self) -> bool:
        return False

    @property
    def portfolio_value(self) -> float:
        total = self._managed_cash
        for symbol, qty in self._managed_positions.items():
            if abs(qty) > 1e-8:
                price = self.fetch_ticker_price(symbol)
                if price is not None:
                    total += qty * price
        return total

    @property
    def all_positions(self) -> dict[str, float]:
        return {s: q for s, q in self._managed_positions.items() if abs(q) > 1e-8}

    @property
    def all_fills(self) -> list[Fill]:
        return list(self._fills)
