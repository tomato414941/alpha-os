"""Hypothesis seeder — register random DSL plus fixed seed hypotheses."""
from __future__ import annotations

import gc
import hashlib
import logging
import random
import signal
import time
from collections import Counter
from dataclasses import dataclass

from alpha_os.config import Config, DATA_DIR
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.universe import build_feature_list, infer_feature_family, stratified_feature_subset
from alpha_os.dsl import temporal_expression_issues, to_string
from alpha_os.dsl.generator import AlphaGenerator
from alpha_os.hypotheses.bootstrap import bootstrap_hypotheses
from alpha_os.hypotheses.identity import (
    expression_feature_families,
    expression_feature_names,
    expression_semantic_key,
)
from alpha_os.hypotheses.store import (
    HypothesisKind,
    HypothesisRecord,
    HypothesisStatus,
    HypothesisStore,
)

logger = logging.getLogger(__name__)

RETIRED_BOOTSTRAP_HYPOTHESES = {
    "technical_volume_price_confirmation": (
        "Retired until the runtime has a real volume-backed implementation."
    ),
    "technical_roc_5_mean_reversion": (
        "Retired because the bootstrap set already keeps a stronger short-horizon mean-reversion seed."
    ),
}

RANDOM_DSL_METADATA = {
    "generator": "hypothesis-seeder",
    "research_quality_source": "exploratory_unscored",
    "research_quality_status": "unscored",
    "registration_stage": "observation_only",
}


@dataclass(frozen=True)
class SeedingRoundStats:
    generated_dsl: int
    inserted_dsl: int
    skipped_dsl: int
    inserted_bootstrap: int
    skipped_bootstrap: int
    elapsed: float


class HypothesisSeederDaemon:
    """Register hypotheses directly into the canonical hypotheses store."""

    def __init__(
        self,
        config: Config,
        *,
        store: HypothesisStore | None = None,
        client=None,
    ):
        self.config = config
        self.generator_cfg = config.alpha_generator
        self.primary_asset = "BTC"
        self._budget = self.generator_cfg.pop_size
        self._round = 0
        self._running = False
        self._client = client if client is not None else build_signal_client_from_config(config.api)
        self._store = store or HypothesisStore(DATA_DIR / "hypotheses.db")

    def run(self) -> None:
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info(
            "HypothesisSeeder started: budget=%d bootstrap=%d",
            self._budget,
            len(bootstrap_hypotheses()),
        )

        try:
            while self._running:
                try:
                    self._run_round()
                    self._round += 1
                except Exception:
                    logger.exception("Round %d failed", self._round)
                    self._sleep(60)
                    continue

                gc.collect()
                if self._running:
                    self._sleep(self.generator_cfg.round_interval)
        finally:
            self.close()

        logger.info("HypothesisSeeder stopped after %d rounds", self._round)

    def close(self) -> None:
        self._store.close()

    def _run_round(self) -> SeedingRoundStats:
        t0 = time.perf_counter()
        seed = int(time.time()) ^ self._round

        all_features = build_feature_list(
            self.primary_asset,
            self._client,
            prefer_cache=True,
        )
        if len(all_features) < self.generator_cfg.min_feature_catalog_size:
            logger.warning(
                "Cached feature catalog too small for seeding: %d < %d; retrying API/catalog refresh",
                len(all_features),
                self.generator_cfg.min_feature_catalog_size,
            )
            all_features = build_feature_list(
                self.primary_asset,
                self._client,
                prefer_cache=False,
                refresh_catalog=True,
            )
        if not all_features:
            logger.warning("No features available, skipping round")
            return SeedingRoundStats(0, 0, 0, 0, 0, 0.0)
        if len(all_features) < self.generator_cfg.min_feature_catalog_size:
            logger.warning(
                "Feature catalog too small after refresh: %d < %d; skipping round",
                len(all_features),
                self.generator_cfg.min_feature_catalog_size,
            )
            return SeedingRoundStats(0, 0, 0, 0, 0, 0.0)

        feature_subset = self._guided_feature_subset(
            all_features,
            k=self.generator_cfg.feature_subset_k,
            seed=seed,
        )
        generator = AlphaGenerator(
            all_features,
            feature_subset=feature_subset or None,
            seed=seed,
        )

        candidates = generator.generate_random(
            self._budget,
            max_depth=self.config.generation.max_depth,
        )
        inserted_dsl, skipped_dsl = self._register_random_dsl(candidates)
        inserted_bootstrap, skipped_bootstrap = self._register_bootstrap_hypotheses()

        elapsed = time.perf_counter() - t0
        stats = SeedingRoundStats(
            generated_dsl=len(candidates),
            inserted_dsl=inserted_dsl,
            skipped_dsl=skipped_dsl,
            inserted_bootstrap=inserted_bootstrap,
            skipped_bootstrap=skipped_bootstrap,
            elapsed=elapsed,
        )
        logger.info(
            "Round %d: dsl generated=%d inserted=%d skipped=%d bootstrap inserted=%d skipped=%d %.1fs",
            self._round,
            stats.generated_dsl,
            stats.inserted_dsl,
            stats.skipped_dsl,
            stats.inserted_bootstrap,
            stats.skipped_bootstrap,
            stats.elapsed,
        )
        return stats

    def _register_random_dsl(self, candidates: list) -> tuple[int, int]:
        inserted = 0
        skipped = 0
        seen_ids: set[str] = set()
        diversity_keys = self._active_random_dsl_diversity_keys()
        for expr in candidates:
            issues = temporal_expression_issues(expr)
            if issues:
                skipped += 1
                logger.debug("Skipping invalid DSL candidate %r: %s", expr, issues[0])
                continue
            expression = to_string(expr)
            if not expression_feature_names(expression):
                skipped += 1
                logger.debug("Skipping featureless DSL candidate %r", expression)
                continue
            hypothesis_id = self._dsl_hypothesis_id(expression)
            if hypothesis_id in seen_ids:
                skipped += 1
                continue
            seen_ids.add(hypothesis_id)
            diversity_key = self._random_dsl_diversity_key(expression)
            if diversity_key is not None and diversity_key in diversity_keys:
                skipped += 1
                continue
            existing = self._store.get(hypothesis_id)
            if existing is not None:
                self._backfill_random_dsl_metadata(existing)
                skipped += 1
                continue

            self._store.register(
                HypothesisRecord(
                    hypothesis_id=hypothesis_id,
                    kind=HypothesisKind.DSL,
                    name=expression[:120],
                    definition={"expression": expression},
                    status=HypothesisStatus.ACTIVE,
                    stake=0.0,
                    source="random_dsl",
                    metadata={
                        **RANDOM_DSL_METADATA,
                        "round": self._round,
                    },
                )
            )
            inserted += 1
        return inserted, skipped

    def _register_bootstrap_hypotheses(self) -> tuple[int, int]:
        self._retire_obsolete_bootstrap_hypotheses()
        inserted = 0
        skipped = 0
        for record in bootstrap_hypotheses():
            existing = self._store.get(record.hypothesis_id)
            if existing is not None:
                self._backfill_bootstrap_metadata(existing, record)
                skipped += 1
                continue
            self._store.register(record)
            inserted += 1
        return inserted, skipped

    def _retire_obsolete_bootstrap_hypotheses(self) -> None:
        for hypothesis_id, reason in RETIRED_BOOTSTRAP_HYPOTHESES.items():
            existing = self._store.get(hypothesis_id)
            if existing is None:
                continue
            metadata = dict(existing.metadata)
            metadata.setdefault("retired_bootstrap_reason", reason)
            metadata.setdefault("retired_bootstrap_at_round", self._round)
            self._store.update_metadata(hypothesis_id, metadata, merge=False)
            self._store.update_stake(hypothesis_id, 0.0)
            self._store.update_status(hypothesis_id, HypothesisStatus.ARCHIVED)

    def _backfill_bootstrap_metadata(
        self,
        existing: HypothesisRecord,
        bootstrap_record: HypothesisRecord,
    ) -> None:
        merged = dict(existing.metadata)
        changed = False
        for key, value in bootstrap_record.metadata.items():
            if key in merged:
                continue
            merged[key] = value
            changed = True
        if changed:
            self._store.update_metadata(existing.hypothesis_id, merged, merge=False)

    def _backfill_random_dsl_metadata(self, existing: HypothesisRecord) -> None:
        if existing.source != "random_dsl":
            return
        merged = dict(existing.metadata)
        changed = False
        for key, value in RANDOM_DSL_METADATA.items():
            if key in merged:
                continue
            merged[key] = value
            changed = True
        if changed:
            self._store.update_metadata(existing.hypothesis_id, merged, merge=False)

    def _active_random_dsl_diversity_keys(self) -> set[tuple[tuple[str, ...], int]]:
        keys: set[tuple[tuple[str, ...], int]] = set()
        for record in self._store.list_observation_active():
            if record.source != "random_dsl":
                continue
            if record.stake <= 0 and not bool(
                record.metadata.get("lifecycle_actionable_live", False)
            ):
                continue
            diversity_key = self._random_dsl_diversity_key(record.expression)
            if diversity_key is not None:
                keys.add(diversity_key)
        return keys

    def _random_dsl_family_success_scores(self) -> dict[str, float]:
        totals: Counter[str] = Counter()
        successes: Counter[str] = Counter()
        for record in self._store.list_observation_active():
            if record.source != "random_dsl":
                continue
            if str(record.metadata.get("research_quality_status", "")) != "scored":
                continue
            families = set(expression_feature_families(record.expression))
            if not families:
                continue
            success = (
                float(record.stake) > 0
                or bool(record.metadata.get("lifecycle_research_retained", False))
                or bool(record.metadata.get("lifecycle_actionable_live", False))
            )
            for family in families:
                totals[family] += 1
                if success:
                    successes[family] += 1
        return {
            family: float(successes.get(family, 0) / count)
            for family, count in totals.items()
            if count > 0
        }

    def _guided_feature_subset(
        self,
        features: list[str],
        *,
        k: int,
        seed: int | None = None,
    ) -> frozenset[str]:
        if k <= 0 or not features:
            return frozenset()
        family_scores = self._random_dsl_family_success_scores()
        if not family_scores:
            return stratified_feature_subset(features, k=k, seed=seed)
        rng = random.Random(seed)
        grouped: dict[str, list[str]] = {}
        for feature in features:
            grouped.setdefault(infer_feature_family(feature), []).append(feature)
        for bucket in grouped.values():
            rng.shuffle(bucket)
        family_order = list(grouped)
        family_order.sort(
            key=lambda family: (
                -family_scores.get(family, 0.5),
                rng.random(),
            )
        )
        selected: list[str] = []
        while len(selected) < min(k, len(features)):
            added = False
            for family in family_order:
                bucket = grouped[family]
                if not bucket:
                    continue
                selected.append(bucket.pop())
                added = True
                if len(selected) >= min(k, len(features)):
                    break
            if not added:
                break
        return frozenset(selected)

    @staticmethod
    def _random_dsl_diversity_key(expression: str) -> tuple[tuple[str, ...], int] | None:
        feature_names = expression_feature_names(expression)
        if not feature_names:
            return None
        feature_families = expression_feature_families(expression)
        if not feature_families:
            return None
        return feature_families, len(feature_names)

    @staticmethod
    def _dsl_hypothesis_id(expression: str) -> str:
        semantic_key = expression_semantic_key(expression)
        digest = hashlib.md5(
            semantic_key.encode(),
            usedforsecurity=False,
        ).hexdigest()[:16]
        return f"dsl_{digest}"

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        self._running = False

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while self._running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))
