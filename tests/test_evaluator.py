"""Tests for alpha evaluator — EvaluationError, evaluate_expression, normalize."""
import numpy as np
import pytest

from alpha_os.alpha.evaluator import (
    FAILED_FITNESS,
    EvaluationError,
    evaluate_alpha,
    evaluate_expression,
    normalize_signal,
)
from alpha_os.dsl.expr import Feature, UnaryOp, Constant


class TestNormalizeSignal:
    def test_basic(self):
        sig = np.array([2.0, -1.0, 3.0, 0.0])
        norm = normalize_signal(sig)
        assert norm.min() >= -1.0
        assert norm.max() <= 1.0

    def test_zero_std_positive(self):
        sig = np.array([5.0, 5.0, 5.0])
        norm = normalize_signal(sig)
        # std=0 → falls back to sign(signal) clipped to [-1,1]
        np.testing.assert_array_equal(norm, [1.0, 1.0, 1.0])

    def test_all_zero(self):
        sig = np.zeros(10)
        norm = normalize_signal(sig)
        np.testing.assert_array_equal(norm, np.zeros(10))


class TestEvaluateExpression:
    def test_feature(self):
        data = {"f1": np.array([1.0, 2.0, 3.0])}
        result = evaluate_expression(Feature("f1"), data, 3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_scalar_broadcast(self):
        data = {"f1": np.array([1.0, 2.0])}
        result = evaluate_expression(Constant(5.0), data, 2)
        np.testing.assert_array_equal(result, [5.0, 5.0])

    def test_nan_replaced(self):
        data = {"f1": np.array([1.0, np.nan, 3.0])}
        result = evaluate_expression(Feature("f1"), data, 3)
        assert np.all(np.isfinite(result))
        assert result[1] == 0.0

    def test_length_mismatch_raises(self):
        data = {"f1": np.array([1.0, 2.0])}
        with pytest.raises(EvaluationError, match="Signal length"):
            evaluate_expression(Feature("f1"), data, 5)

    def test_missing_feature_raises(self):
        data = {"f1": np.array([1.0])}
        with pytest.raises(EvaluationError):
            evaluate_expression(Feature("f_missing"), data, 1)


class TestEvaluateAlpha:
    def test_roundtrip(self):
        data = {"f1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        result = evaluate_alpha("f1", data, 5, normalize=True)
        assert result.shape == (5,)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_no_normalize(self):
        data = {"f1": np.array([10.0, 20.0, 30.0])}
        result = evaluate_alpha("f1", data, 3, normalize=False)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])


class TestFailedFitness:
    def test_sentinel_value(self):
        assert FAILED_FITNESS < -900
        assert isinstance(FAILED_FITNESS, float)

    def test_evaluation_error_is_exception(self):
        with pytest.raises(EvaluationError):
            raise EvaluationError("test failure")
