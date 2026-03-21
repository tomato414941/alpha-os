"""Registry producer — evaluates all active alphas and writes predictions.

This is the bridge between the legacy registry (DSL expressions) and the
prediction store. It runs daily before the trader, ensuring all active
signals have fresh predictions in the store.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import numpy as np

from ..alpha.evaluator import sanitize_signal
from ..alpha.managed_alphas import ManagedAlphaStore
from ..config import Config, DATA_DIR, asset_data_dir
from ..data.signal_client import build_signal_client_from_config
from ..data.store import DataStore
from ..data.universe import build_feature_list, price_signal
from ..dsl import parse
from .store import Prediction, PredictionStore, SignalMeta

logger = logging.getLogger(__name__)


def produce_daily_predictions(
    asset: str,
    config: Config,
    *,
    today: str | None = None,
    prediction_store: PredictionStore | None = None,
) -> int:
    """Evaluate all active alphas and write predictions to the store.

    Returns the number of predictions written.
    """
    t0 = time.perf_counter()
    today = today or date.today().isoformat()

    # Load data
    client = build_signal_client_from_config(config.api)
    features = build_feature_list(asset, client)
    db_path = DATA_DIR / "alpha_cache.db"
    store = DataStore(db_path, client)
    try:
        if client.health():
            store.sync(features)
    except Exception as exc:
        logger.warning("API sync failed: %s", exc)

    matrix = store.get_matrix(features)
    store.close()

    ps = price_signal(asset)
    if ps in matrix.columns:
        matrix = matrix[matrix[ps].notna()]
    matrix = matrix.fillna(0)
    data = {col: matrix[col].values for col in matrix.columns}

    # Load active alphas
    registry = ManagedAlphaStore(asset_data_dir(asset) / "alpha_registry.db")
    active = [r for r in registry.list_all() if r.stake > 0]
    registry.close()

    if not active:
        logger.info("No active alphas with stake > 0")
        return 0

    # Initialize prediction store
    pred_store = prediction_store or PredictionStore()
    n_days = len(matrix)

    # Evaluate each alpha
    predictions: list[Prediction] = []
    n_failed = 0

    for record in active:
        try:
            expr = parse(record.expression)
            sig = sanitize_signal(expr.evaluate(data))
            if sig.ndim == 0:
                sig = np.full(n_days, float(sig))
            if len(sig) == 0:
                continue

            value = float(sig[-1])
            if not np.isfinite(value):
                continue

            # Register signal metadata (idempotent)
            signal_id = record.alpha_id
            pred_store.register_signal(SignalMeta(
                signal_id=signal_id,
                source="registry",
                definition=record.expression,
                horizon=1,
            ))

            predictions.append(Prediction(
                signal_id=signal_id,
                date=today,
                asset=asset,
                value=value,
                horizon=1,
            ))
        except Exception:
            n_failed += 1
            continue

    n_written = pred_store.write(predictions) if predictions else 0

    if prediction_store is None:
        pred_store.close()

    elapsed = time.perf_counter() - t0
    logger.info(
        "Registry producer: %d alphas -> %d predictions written "
        "(%d failed), %.1fs",
        len(active), n_written, n_failed, elapsed,
    )
    return n_written
