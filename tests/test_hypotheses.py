from alpha_os.hypotheses import (
    HypothesisContribution,
    HypothesisKind,
    HypothesisRecord,
    HypothesisStatus,
    HypothesisStore,
)


class TestHypothesisStore:
    def _make_store(self, tmp_path):
        return HypothesisStore(db_path=tmp_path / "hypotheses.db")

    def test_register_and_get_dsl_hypothesis(self, tmp_path):
        store = self._make_store(tmp_path)
        record = HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            name="fear-greed spread",
            definition={"expression": "(sub fear_greed dxy)"},
            stake=1.25,
            source="random_dsl",
            metadata={"tag": "seed"},
        )

        store.register(record)
        got = store.get("h1")

        assert got is not None
        assert got.kind == HypothesisKind.DSL
        assert got.definition["expression"] == "(sub fear_greed dxy)"
        assert got.horizon == "20D2L"
        assert got.target_kind == "forward_residual_return"
        assert got.scope == {"universe": "core_universe_1000"}
        assert got.metadata["tag"] == "seed"
        store.close()

    def test_register_and_get_non_dsl_hypothesis(self, tmp_path):
        store = self._make_store(tmp_path)
        record = HypothesisRecord(
            hypothesis_id="h-tech-1",
            kind=HypothesisKind.TECHNICAL,
            name="rsi reversion",
            definition={
                "indicator": "rsi_reversion",
                "params": {"window": 14, "threshold": 30},
                "inputs": ["btc_ohlcv"],
            },
            source="technical_library",
        )

        store.register(record)
        got = store.get("h-tech-1")

        assert got is not None
        assert got.kind == HypothesisKind.TECHNICAL
        assert got.definition["indicator"] == "rsi_reversion"
        store.close()

    def test_list_active_returns_only_positive_stake_active_rows(self, tmp_path):
        store = self._make_store(tmp_path)
        store.register(
            HypothesisRecord(
                hypothesis_id="h1",
                kind=HypothesisKind.DSL,
                definition={"expression": "f1"},
                stake=1.0,
            )
        )
        store.register(
            HypothesisRecord(
                hypothesis_id="h2",
                kind=HypothesisKind.MANUAL,
                definition={"rule": "wait"},
                stake=0.0,
            )
        )
        store.register(
            HypothesisRecord(
                hypothesis_id="h3",
                kind=HypothesisKind.EXTERNAL,
                definition={"provider": "x"},
                status=HypothesisStatus.PAUSED,
                stake=2.0,
            )
        )

        active = store.list_active()

        assert [row.hypothesis_id for row in active] == ["h1"]
        store.close()

    def test_update_status_and_stake(self, tmp_path):
        store = self._make_store(tmp_path)
        store.register(
            HypothesisRecord(
                hypothesis_id="h1",
                kind=HypothesisKind.ML,
                definition={"model_ref": "models/m1.json"},
                status=HypothesisStatus.PAUSED,
                stake=0.0,
            )
        )

        store.update_status("h1", HypothesisStatus.ACTIVE)
        store.update_stake("h1", 3.5)
        got = store.get("h1")

        assert got is not None
        assert got.status == HypothesisStatus.ACTIVE
        assert got.stake == 3.5
        store.close()

    def test_update_metadata_merges_by_default(self, tmp_path):
        store = self._make_store(tmp_path)
        store.register(
            HypothesisRecord(
                hypothesis_id="h1",
                kind=HypothesisKind.DSL,
                definition={"expression": "f1"},
                metadata={"generator": "seed", "round": 1},
            )
        )

        store.update_metadata("h1", {"invalid_reason": "requires_series_input"})
        got = store.get("h1")

        assert got is not None
        assert got.metadata == {
            "generator": "seed",
            "round": 1,
            "invalid_reason": "requires_series_input",
        }
        store.close()

    def test_count_by_status(self, tmp_path):
        store = self._make_store(tmp_path)
        store.register(
            HypothesisRecord(
                hypothesis_id="h1",
                kind=HypothesisKind.DSL,
                definition={"expression": "f1"},
            )
        )
        store.register(
            HypothesisRecord(
                hypothesis_id="h2",
                kind=HypothesisKind.MANUAL,
                definition={"rule": "x"},
                status=HypothesisStatus.PAUSED,
            )
        )

        assert store.count() == 2
        assert store.count(status=HypothesisStatus.ACTIVE) == 1
        assert store.count(status=HypothesisStatus.PAUSED) == 1
        store.close()

    def test_runtime_compatibility_aliases(self, tmp_path):
        store = self._make_store(tmp_path)
        store.register(
            HypothesisRecord(
                hypothesis_id="h1",
                kind=HypothesisKind.DSL,
                definition={"expression": "(sub fear_greed dxy)"},
                metadata={"oos_sharpe": 1.5, "oos_log_growth": 0.3},
                stake=2.0,
            )
        )

        record = store.get("h1")

        assert record is not None
        assert record.hypothesis_id == "h1"
        assert record.expression == "(sub fear_greed dxy)"
        assert record.oos_fitness("sharpe") == 1.5
        assert record.oos_fitness("log_growth") == 0.3
        store.close()

    def test_top_by_stake_matches_live_runtime_selection(self, tmp_path):
        store = self._make_store(tmp_path)
        store.register(
            HypothesisRecord(
                hypothesis_id="h1",
                kind=HypothesisKind.DSL,
                definition={"expression": "f1"},
                stake=1.0,
            )
        )
        store.register(
            HypothesisRecord(
                hypothesis_id="h2",
                kind=HypothesisKind.ML,
                definition={"model_ref": "m1"},
                stake=3.0,
            )
        )
        store.register(
            HypothesisRecord(
                hypothesis_id="h3",
                kind=HypothesisKind.MANUAL,
                definition={"rule": "pause"},
                status=HypothesisStatus.PAUSED,
                stake=4.0,
            )
        )

        assert [row.hypothesis_id for row in store.top_by_stake(2)] == ["h2", "h1"]
        assert [row.hypothesis_id for row in store.list_live()] == ["h2", "h1"]
        store.close()

    def test_record_and_list_contributions(self, tmp_path):
        store = self._make_store(tmp_path)
        store.register(
            HypothesisRecord(
                hypothesis_id="h1",
                kind=HypothesisKind.DSL,
                definition={"expression": "f1"},
                stake=1.0,
            )
        )

        store.record_contribution("h1", date="2026-03-20", contribution=0.12)
        store.record_contribution("h1", date="2026-03-21", contribution=-0.03)

        rows = store.list_contributions("h1")

        assert rows == [
            HypothesisContribution(
                hypothesis_id="h1",
                date="2026-03-21",
                contribution=-0.03,
                created_at=rows[0].created_at,
            ),
            HypothesisContribution(
                hypothesis_id="h1",
                date="2026-03-20",
                contribution=0.12,
                created_at=rows[1].created_at,
            ),
        ]
        assert store.contribution_history("h1") == [-0.03, 0.12]
        store.close()

    def test_record_contribution_replaces_same_day_value(self, tmp_path):
        store = self._make_store(tmp_path)
        store.register(
            HypothesisRecord(
                hypothesis_id="h1",
                kind=HypothesisKind.ML,
                definition={"model_ref": "m1"},
                stake=1.0,
            )
        )

        store.record_contribution("h1", date="2026-03-21", contribution=0.1)
        store.record_contribution("h1", date="2026-03-21", contribution=0.25)

        rows = store.list_contributions("h1")

        assert len(rows) == 1
        assert rows[0].contribution == 0.25
        store.close()
