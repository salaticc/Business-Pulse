import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate import save_results
from src.data import load_daily
from src.evaluation import EvaluationConfig, METHODS, evaluate, expanding_folds
from src.model import FEATURES, add_features, metrics, new_model


@pytest.fixture
def daily():
    return load_daily("data/demo/daily_sales.csv")[0]


def test_actual_feature_values_match_strict_past():
    values = np.arange(1., 101.)
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=100), "revenue": values})
    row = add_features(frame).iloc[22]  # raw position 50 after 28 warmup rows
    for lag in (1, 7, 14, 21, 28):
        assert row[f"lag_{lag}"] == values[50 - lag]
    assert row["mean_7"] == pytest.approx(np.mean(values[43:50]))
    assert row["mean_28"] == pytest.approx(np.mean(values[22:50]))
    assert row["std_28"] == pytest.approx(np.std(values[22:50], ddof=1))
    assert row["baseline"] == np.median([values[50-lag] for lag in (7, 14, 21, 28)])


def test_target_orders_aov_and_future_cannot_change_features_at_prediction(daily):
    changed = daily.copy()
    origin = 150
    changed.loc[origin:, "revenue"] += 1_000_000
    changed.loc[:, "orders"] = 99999
    changed.loc[:, "aov"] = 99999.
    a, b = add_features(daily), add_features(changed)
    mask = a.date.le(daily.date.iloc[origin])
    pd.testing.assert_frame_equal(a.loc[mask, FEATURES + ["baseline"]], b.loc[mask, FEATURES + ["baseline"]])


def test_expanding_boundaries_and_identical_validation_rows(daily):
    cv, test, report = evaluate(daily)
    assert len(cv) == 84 and len(test) == 60
    assert cv.date.is_unique and cv.date.max() < test.date.min()
    assert [f["n_train"] for f in report["folds"]] == [108, 136, 164]
    for fold in report["folds"]:
        assert fold["train_dates"][1] < fold["validation_dates"][0]
        assert fold["validation_dates"][1] < report["test_dates"][0]
        window = cv[cv.fold.eq(fold["fold"])]
        assert len(window) == 28
        for name in METHODS:
            assert np.isfinite(window[name]).all() and window[name].ge(0).all()
            assert fold["metrics"][name] == metrics(window.revenue, window[name])
    pooled = report["cv_pooled_metrics"]
    assert report["selected_method"] == min(METHODS, key=lambda n: pooled[n]["wape_pct"])
    pd.testing.assert_series_equal(test.chosen_forecast, test[report["selected_method"]], check_names=False)
    for name in METHODS:
        scores = [f["metrics"][name]["mae"] for f in report["folds"]]
        spread = report["cv_window_dispersion"][name]["mae"]
        assert spread["mean"] == pytest.approx(np.mean(scores))
        assert spread["std_population"] == pytest.approx(np.std(scores))


def test_holdout_mutation_does_not_affect_selection_cv_or_earlier_predictions(daily):
    cv_a, test_a, a = evaluate(daily)
    changed = daily.copy()
    changed.loc[len(changed)-60:, "revenue"] += 1_000_000
    cv_b, test_b, b = evaluate(changed)
    pd.testing.assert_frame_equal(cv_a, cv_b)
    for field in ("selected_method", "folds", "cv_pooled_metrics", "cv_window_dispersion", "reproducibility"):
        assert a[field] == b[field]
    # The first test forecast cannot know any test target. Later days may use them as lags.
    pd.testing.assert_series_equal(test_a[list(METHODS)].iloc[0], test_b[list(METHODS)].iloc[0])
    assert a["test_metrics"] != b["test_metrics"]
    changed = daily.copy()
    changed.loc[len(changed)-30:, "revenue"] += 1_000_000
    _, test_c, _ = evaluate(changed)
    pd.testing.assert_frame_equal(test_a.iloc[:31][list(METHODS)], test_c.iloc[:31][list(METHODS)])


def test_fit_never_receives_window_or_test_targets(monkeypatch, daily):
    import src.evaluation as module
    original = module.new_model
    fits = []

    def factory():
        model = original()
        fit = model.fit

        def checked_fit(x, y):
            fits.append((x.index.min(), x.index.max(), len(y)))
            return fit(x, y)

        model.fit = checked_fit
        return model

    monkeypatch.setattr(module, "new_model", factory)
    evaluate(daily)
    assert fits == [(0, 107, 108), (0, 135, 136), (0, 163, 164), (0, 191, 192)]


def test_validation_only_does_not_predict_test(monkeypatch, daily):
    import src.evaluation as module
    predict = module._predict
    origins = []

    def checked(train, window):
        origins.append(window.date.min())
        assert window.date.max() < daily.date.iloc[-60]
        return predict(train, window)

    monkeypatch.setattr(module, "_predict", checked)
    _, test, report = evaluate(daily, evaluate_test=False)
    assert len(origins) == 3 and test is None
    assert not report["test_evaluated"] and "test_metrics" not in report


def test_minimum_length_is_explicit_and_no_windows_overlap():
    config = EvaluationConfig()
    assert expanding_folds(144, config)[0].train_end == 60
    with pytest.raises(ValueError, match="232"):
        expanding_folds(143, config)
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=232), "revenue": np.ones(232)})
    cv, test, report = evaluate(frame)
    assert len(cv) == 84 and len(test) == 60
    assert report["selected_method"] == METHODS[0]  # deterministic simple-method tie break
    with pytest.raises(ValueError, match="232"):
        evaluate(frame.iloc[1:])


@pytest.mark.parametrize("kwargs", [{"n_splits": 1}, {"n_splits": True}, {"validation_days": 6},
                                  {"test_days": 0}, {"min_train_days": 59}, {"n_splits": 2.5}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)


def test_metrics_do_not_round_or_fake_zero_denominator():
    score = metrics([0.1, 0.2], [0.2, 0.4])
    assert score["wape_pct"] == pytest.approx(100.)
    assert score["mae"] == pytest.approx(0.15)
    assert score["rmse"] == pytest.approx(np.sqrt(0.025))
    assert metrics([3.], [2.])["wape_pct"] == pytest.approx(100/3)
    assert metrics([0., 0.], [1., 2.])["wape_pct"] is None


@pytest.mark.parametrize("y,p", [([], []), ([1], [1, 2]), ([np.nan], [1]), ([[1]], [[1]])])
def test_invalid_metrics_rejected(y, p):
    with pytest.raises(ValueError):
        metrics(y, p)


def test_zero_revenue_cv_requires_another_selection_metric(daily):
    daily["revenue"] = 0.
    with pytest.raises(ValueError, match="WAPE"):
        evaluate(daily)


def test_zero_fold_and_zero_test_report_null_not_nan(daily, tmp_path):
    daily.loc[len(daily)-88:, "revenue"] = 0.
    cv, test, report = evaluate(daily)
    assert report["folds"][-1]["metrics"][METHODS[0]]["wape_pct"] is None
    assert report["cv_window_dispersion"][METHODS[0]]["wape_pct"]["defined_windows"] == 2
    assert report["selected_test_metrics"]["wape_pct"] is None
    save_results(cv, test, report, tmp_path)
    text = (tmp_path / "evaluation_report.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["test_metrics"][METHODS[0]]["wape_pct"] is None


def test_model_has_no_implicit_random_validation():
    assert new_model().early_stopping is False
    assert new_model().random_state == 42


def test_repeated_experiment_and_artifacts_are_identical(daily, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    a_cv, a_test, a = evaluate(daily)
    b_cv, b_test, b = evaluate(daily)
    pd.testing.assert_frame_equal(a_cv, b_cv)
    pd.testing.assert_frame_equal(a_test, b_test)
    assert a == b
    save_results(a_cv, a_test, a, first)
    save_results(b_cv, b_test, b, second)
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name
    save_results(a_cv, None, {**a, "test_evaluated": False}, first)
    assert not (first / "final_test_predictions.csv").exists()


def test_cli_validation_only_and_manifest(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.evaluate", "--validation-only", "--output", str(tmp_path)],
        capture_output=True, check=False, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    assert not report["test_evaluated"] and "test_metrics" not in report
    assert len(report["reproducibility"]["input_sha256"]) == 64
    assert "src/evaluation.py" in report["reproducibility"]["code_sha256"]
    assert (tmp_path / "evaluation.html").exists()
