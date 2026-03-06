"""Tests for two-level ensemble aggregation (Path B)."""
import numpy as np
import pytest

from alpha_os.voting.ensemble import (
    ensemble_sizing,
    compute_cell_long_pcts,
)


class TestEnsembleSizing:
    def test_strong_consensus_long(self):
        # All cells vote ~90% long
        pcts = [0.9, 0.85, 0.95, 0.88, 0.92]
        result = ensemble_sizing(pcts)
        assert result.direction == 1.0
        assert result.confidence > 0.5
        assert result.skew_adj > 0.8

    def test_strong_consensus_short(self):
        pcts = [0.1, 0.15, 0.05, 0.12, 0.08]
        result = ensemble_sizing(pcts)
        assert result.direction == -1.0
        assert result.confidence > 0.5

    def test_split_vote(self):
        # 50/50 split
        pcts = [0.9, 0.1, 0.8, 0.2, 0.5]
        result = ensemble_sizing(pcts)
        assert result.confidence < 0.3
        assert result.sigma_cells > 0.2

    def test_unanimous(self):
        pcts = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = ensemble_sizing(pcts)
        assert result.direction == 1.0
        assert result.confidence == 1.0
        assert result.sigma_cells == 0.0

    def test_below_min_cells(self):
        result = ensemble_sizing([0.9, 0.8], min_cells=5)
        assert result.direction == 0.0
        assert result.confidence == 0.0

    def test_skew_penalty(self):
        # Skewed distribution: most cells say long, one says strong short
        pcts = [0.9, 0.85, 0.88, 0.92, 0.1]
        result = ensemble_sizing(pcts, skew_k=0.5)
        assert result.skew_adj < 1.0
        # Compare with no-skew case
        no_skew = ensemble_sizing([0.9, 0.85, 0.88, 0.92, 0.87], skew_k=0.5)
        assert result.skew_adj < no_skew.skew_adj

    def test_n_cells_reported(self):
        pcts = [0.5] * 10
        result = ensemble_sizing(pcts)
        assert result.n_cells == 10

    def test_mu_sigma_reported(self):
        pcts = [0.6, 0.7, 0.8, 0.9, 0.5]
        result = ensemble_sizing(pcts)
        assert result.mu_cells == pytest.approx(np.mean(pcts))
        assert result.sigma_cells == pytest.approx(np.std(pcts))


class TestComputeCellLongPcts:
    def test_basic(self):
        signals = {
            (0, 0, 0): [1.0, -1.0, 1.0],   # 2/3 long
            (1, 0, 0): [-1.0, -1.0],         # 0/2 long
            (2, 0, 0): [1.0, 1.0, 1.0, 1.0], # 4/4 long
        }
        pcts = compute_cell_long_pcts(None, signals)
        assert len(pcts) == 3
        assert sorted(pcts) == pytest.approx([0.0, 2 / 3, 1.0])

    def test_empty_cell_skipped(self):
        signals = {
            (0, 0, 0): [1.0],
            (1, 0, 0): [],
        }
        pcts = compute_cell_long_pcts(None, signals)
        assert len(pcts) == 1

    def test_single_signal_per_cell(self):
        signals = {
            (0, 0, 0): [1.0],
            (1, 0, 0): [-1.0],
        }
        pcts = compute_cell_long_pcts(None, signals)
        assert sorted(pcts) == [0.0, 1.0]
