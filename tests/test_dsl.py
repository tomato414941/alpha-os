import numpy as np
import pytest

from alpha_os.dsl import (
    Feature,
    Constant,
    UnaryOp,
    BinaryOp,
    RollingOp,
    PairRollingOp,
    ConditionalOp,
    LagOp,
    parse,
    to_string,
    AlphaGenerator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 100
    return {
        "nvda": rng.standard_normal(n).cumsum(),
        "sp500": rng.standard_normal(n).cumsum(),
        "vix": rng.standard_normal(n).cumsum() + 20,
    }


# ---------------------------------------------------------------------------
# Expression node construction & repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_feature(self):
        assert repr(Feature("nvda")) == "nvda"

    def test_constant(self):
        assert repr(Constant(1.5)) == "1.5"

    def test_unary(self):
        assert repr(UnaryOp("neg", Feature("x"))) == "(neg x)"

    def test_binary(self):
        assert repr(BinaryOp("add", Feature("x"), Feature("y"))) == "(add x y)"

    def test_rolling(self):
        assert repr(RollingOp("mean", 20, Feature("x"))) == "(mean_20 x)"

    def test_pair_rolling(self):
        e = PairRollingOp("corr", 60, Feature("x"), Feature("y"))
        assert repr(e) == "(corr_60 x y)"

    def test_nested(self):
        e = BinaryOp(
            "sub",
            RollingOp("roc", 10, Feature("nvda")),
            RollingOp("roc", 10, Feature("sp500")),
        )
        assert repr(e) == "(sub (roc_10 nvda) (roc_10 sp500))"

    def test_conditional(self):
        e = ConditionalOp("if_gt", Feature("x"), Feature("y"), Constant(1.0), Constant(-1.0))
        assert repr(e) == "(if_gt x y 1.0 -1.0)"

    def test_lag(self):
        assert repr(LagOp("lag", 5, Feature("x"))) == "(lag_5 x)"


# ---------------------------------------------------------------------------
# Parser round-trip
# ---------------------------------------------------------------------------

class TestParser:
    @pytest.mark.parametrize(
        "s",
        [
            "nvda",
            "1.5",
            "(neg nvda)",
            "(abs sp500)",
            "(add nvda sp500)",
            "(sub (roc_10 nvda) (roc_10 sp500))",
            "(corr_60 nvda sp500)",
            "(neg (sub (rank_20 nvda) 0.5))",
            "(div nvda sp500)",
            "(ema_30 nvda)",
            "(lag_10 nvda)",
            "(lag_5 (roc_10 nvda))",
            "(if_gt nvda sp500 1.0 -1.0)",
            "(if_gt (mean_20 nvda) 0.5 nvda (neg sp500))",
        ],
    )
    def test_round_trip(self, s):
        expr = parse(s)
        assert to_string(expr) == s

    def test_parse_feature(self):
        e = parse("nvda")
        assert isinstance(e, Feature)
        assert e.name == "nvda"

    def test_parse_constant(self):
        e = parse("2.0")
        assert isinstance(e, Constant)
        assert e.value == 2.0

    def test_parse_unary(self):
        e = parse("(neg nvda)")
        assert isinstance(e, UnaryOp)
        assert e.op == "neg"
        assert isinstance(e.child, Feature)

    def test_parse_binary(self):
        e = parse("(add nvda sp500)")
        assert isinstance(e, BinaryOp)
        assert e.op == "add"

    def test_parse_rolling(self):
        e = parse("(mean_20 nvda)")
        assert isinstance(e, RollingOp)
        assert e.op == "mean"
        assert e.window == 20

    def test_parse_pair_rolling(self):
        e = parse("(corr_60 nvda sp500)")
        assert isinstance(e, PairRollingOp)
        assert e.op == "corr"
        assert e.window == 60

    def test_parse_lag(self):
        e = parse("(lag_10 nvda)")
        assert isinstance(e, LagOp)
        assert e.op == "lag"
        assert e.window == 10

    def test_parse_conditional(self):
        e = parse("(if_gt nvda sp500 1.0 -1.0)")
        assert isinstance(e, ConditionalOp)
        assert e.op == "if_gt"
        assert isinstance(e.condition_left, Feature)
        assert isinstance(e.then_branch, Constant)

    def test_parse_error_empty(self):
        with pytest.raises(SyntaxError):
            parse("")

    def test_parse_error_extra_tokens(self):
        with pytest.raises(SyntaxError):
            parse("nvda sp500")

    def test_parse_error_unknown_op(self):
        with pytest.raises(SyntaxError):
            parse("(unknown nvda)")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_feature_eval(self, sample_data):
        result = Feature("nvda").evaluate(sample_data)
        np.testing.assert_array_equal(result, sample_data["nvda"])

    def test_feature_missing(self, sample_data):
        with pytest.raises(KeyError):
            Feature("missing").evaluate(sample_data)

    def test_constant_eval(self, sample_data):
        result = Constant(3.14).evaluate(sample_data)
        assert result == 3.14

    def test_neg(self, sample_data):
        result = UnaryOp("neg", Feature("nvda")).evaluate(sample_data)
        np.testing.assert_array_equal(result, -sample_data["nvda"])

    def test_abs(self, sample_data):
        result = UnaryOp("abs", Feature("nvda")).evaluate(sample_data)
        np.testing.assert_array_equal(result, np.abs(sample_data["nvda"]))

    def test_sign(self, sample_data):
        result = UnaryOp("sign", Feature("nvda")).evaluate(sample_data)
        np.testing.assert_array_equal(result, np.sign(sample_data["nvda"]))

    def test_log_safe(self, sample_data):
        result = UnaryOp("log", Feature("nvda")).evaluate(sample_data)
        expected = np.sign(sample_data["nvda"]) * np.log1p(np.abs(sample_data["nvda"]))
        np.testing.assert_array_almost_equal(result, expected)

    def test_zscore(self, sample_data):
        result = UnaryOp("zscore", Feature("nvda")).evaluate(sample_data)
        x = sample_data["nvda"]
        expected = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-10)
        np.testing.assert_array_almost_equal(result, expected)

    def test_add(self, sample_data):
        result = BinaryOp("add", Feature("nvda"), Feature("sp500")).evaluate(sample_data)
        expected = sample_data["nvda"] + sample_data["sp500"]
        np.testing.assert_array_equal(result, expected)

    def test_sub(self, sample_data):
        result = BinaryOp("sub", Feature("nvda"), Feature("sp500")).evaluate(sample_data)
        expected = sample_data["nvda"] - sample_data["sp500"]
        np.testing.assert_array_equal(result, expected)

    def test_mul(self, sample_data):
        result = BinaryOp("mul", Feature("nvda"), Feature("sp500")).evaluate(sample_data)
        expected = sample_data["nvda"] * sample_data["sp500"]
        np.testing.assert_array_equal(result, expected)

    def test_div_safe(self, sample_data):
        result = BinaryOp("div", Feature("nvda"), Feature("sp500")).evaluate(sample_data)
        assert np.all(np.isfinite(result))

    def test_max(self, sample_data):
        result = BinaryOp("max", Feature("nvda"), Feature("sp500")).evaluate(sample_data)
        expected = np.maximum(sample_data["nvda"], sample_data["sp500"])
        np.testing.assert_array_equal(result, expected)

    def test_min(self, sample_data):
        result = BinaryOp("min", Feature("nvda"), Feature("sp500")).evaluate(sample_data)
        expected = np.minimum(sample_data["nvda"], sample_data["sp500"])
        np.testing.assert_array_equal(result, expected)

    def test_rolling_mean(self, sample_data):
        result = RollingOp("mean", 10, Feature("nvda")).evaluate(sample_data)
        assert np.all(np.isnan(result[:9]))
        assert np.all(np.isfinite(result[9:]))

    def test_rolling_std(self, sample_data):
        result = RollingOp("std", 10, Feature("nvda")).evaluate(sample_data)
        assert np.all(np.isnan(result[:9]))
        assert np.all(np.isfinite(result[9:]))

    def test_rolling_ts_max(self, sample_data):
        result = RollingOp("ts_max", 5, Feature("nvda")).evaluate(sample_data)
        assert np.all(np.isnan(result[:4]))
        # At each valid point, ts_max >= current value
        valid = result[4:]
        actual = sample_data["nvda"][4:]
        assert np.all(valid >= actual - 1e-10)

    def test_rolling_ts_min(self, sample_data):
        result = RollingOp("ts_min", 5, Feature("nvda")).evaluate(sample_data)
        assert np.all(np.isnan(result[:4]))
        valid = result[4:]
        actual = sample_data["nvda"][4:]
        assert np.all(valid <= actual + 1e-10)

    def test_delta(self, sample_data):
        result = RollingOp("delta", 5, Feature("nvda")).evaluate(sample_data)
        assert np.all(np.isnan(result[:5]))
        x = sample_data["nvda"]
        np.testing.assert_array_almost_equal(result[5:], x[5:] - x[:-5])

    def test_roc(self, sample_data):
        result = RollingOp("roc", 5, Feature("nvda")).evaluate(sample_data)
        assert np.all(np.isnan(result[:5]))
        assert np.all(np.isfinite(result[5:]))

    def test_rank(self, sample_data):
        result = RollingOp("rank", 10, Feature("nvda")).evaluate(sample_data)
        assert np.all(np.isnan(result[:9]))
        valid = result[9:]
        assert np.all((valid >= 0) & (valid <= 1))

    def test_ema(self, sample_data):
        result = RollingOp("ema", 10, Feature("nvda")).evaluate(sample_data)
        # EMA has no warmup NaN in this implementation (adjust=False)
        assert np.all(np.isfinite(result))

    def test_corr(self, sample_data):
        result = PairRollingOp("corr", 20, Feature("nvda"), Feature("sp500")).evaluate(
            sample_data
        )
        assert np.all(np.isnan(result[:19]))
        valid = result[19:]
        assert np.all((valid >= -1 - 1e-10) & (valid <= 1 + 1e-10))

    def test_cov(self, sample_data):
        result = PairRollingOp("cov", 20, Feature("nvda"), Feature("sp500")).evaluate(
            sample_data
        )
        assert np.all(np.isnan(result[:19]))

    def test_lag(self, sample_data):
        result = LagOp("lag", 10, Feature("nvda")).evaluate(sample_data)
        assert result.shape == (100,)
        assert np.all(np.isnan(result[:10]))
        np.testing.assert_array_equal(result[10:], sample_data["nvda"][:-10])

    def test_conditional(self, sample_data):
        expr = ConditionalOp(
            "if_gt", Feature("nvda"), Feature("sp500"), Constant(100.0), Constant(-100.0)
        )
        result = expr.evaluate(sample_data)
        assert result.shape == (100,)
        expected = np.where(
            sample_data["nvda"] > sample_data["sp500"], 100.0, -100.0
        )
        np.testing.assert_array_equal(result, expected)

    def test_nested_expression(self, sample_data):
        # (sub (roc_10 nvda) (roc_10 sp500))
        expr = BinaryOp(
            "sub",
            RollingOp("roc", 10, Feature("nvda")),
            RollingOp("roc", 10, Feature("sp500")),
        )
        result = expr.evaluate(sample_data)
        assert result.shape == (100,)

    def test_parse_and_evaluate(self, sample_data):
        expr = parse("(sub (mean_10 nvda) (mean_10 sp500))")
        result = expr.evaluate(sample_data)
        assert result.shape == (100,)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unknown_unary_op(self):
        with pytest.raises(ValueError):
            UnaryOp("bad_op", Feature("x"))

    def test_unknown_binary_op(self):
        with pytest.raises(ValueError):
            BinaryOp("bad_op", Feature("x"), Feature("y"))

    def test_unknown_rolling_op(self):
        with pytest.raises(ValueError):
            RollingOp("bad_op", 10, Feature("x"))

    def test_unknown_pair_rolling_op(self):
        with pytest.raises(ValueError):
            PairRollingOp("bad_op", 10, Feature("x"), Feature("y"))

    def test_unknown_conditional_op(self):
        with pytest.raises(ValueError):
            ConditionalOp("bad_op", Feature("x"), Feature("y"), Feature("z"), Feature("w"))

    def test_unknown_lag_op(self):
        with pytest.raises(ValueError):
            LagOp("bad_op", 10, Feature("x"))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_generate_random_count(self):
        gen = AlphaGenerator(["nvda", "sp500", "vix"], seed=42)
        exprs = gen.generate_random(10, max_depth=3)
        assert len(exprs) == 10

    def test_generate_random_deterministic(self):
        gen1 = AlphaGenerator(["nvda", "sp500"], seed=123)
        gen2 = AlphaGenerator(["nvda", "sp500"], seed=123)
        r1 = [repr(e) for e in gen1.generate_random(5)]
        r2 = [repr(e) for e in gen2.generate_random(5)]
        assert r1 == r2

    def test_generate_random_parseable(self):
        gen = AlphaGenerator(["a", "b", "c"], seed=0)
        for expr in gen.generate_random(20, max_depth=3):
            s = to_string(expr)
            roundtrip = parse(s)
            assert to_string(roundtrip) == s

    def test_generate_random_evaluable(self, sample_data):
        gen = AlphaGenerator(["nvda", "sp500", "vix"], seed=7)
        for expr in gen.generate_random(10, max_depth=2):
            result = expr.evaluate(sample_data)
            assert isinstance(result, np.ndarray) or isinstance(result, np.floating)

    def test_generate_from_templates(self):
        gen = AlphaGenerator(["nvda", "sp500", "vix"])
        templates = gen.generate_from_templates()
        assert len(templates) > 0
        for expr in templates:
            s = to_string(expr)
            roundtrip = parse(s)
            assert to_string(roundtrip) == s

    def test_generate_from_templates_evaluable(self, sample_data):
        gen = AlphaGenerator(["nvda", "sp500", "vix"])
        for expr in gen.generate_from_templates():
            result = expr.evaluate(sample_data)
            assert isinstance(result, np.ndarray) or isinstance(result, np.floating)

    def test_mutate(self):
        gen = AlphaGenerator(["nvda", "sp500", "vix"], seed=42)
        original = parse("(sub (roc_10 nvda) (roc_10 sp500))")
        original_str = to_string(original)
        # Run multiple mutations; at least one should differ
        mutated_strs = set()
        for _ in range(50):
            m = gen.mutate(original)
            mutated_strs.add(to_string(m))
        # Original should be unchanged
        assert to_string(original) == original_str
        # At least some mutations should produce different expressions
        assert len(mutated_strs) > 1

    def test_mutate_preserves_structure(self, sample_data):
        gen = AlphaGenerator(["nvda", "sp500", "vix"], seed=99)
        original = parse("(neg (mean_20 nvda))")
        for _ in range(20):
            m = gen.mutate(original)
            # Should still parse and evaluate
            s = to_string(m)
            roundtrip = parse(s)
            assert to_string(roundtrip) == s

    def test_empty_features_raises(self):
        with pytest.raises(ValueError):
            AlphaGenerator([])
