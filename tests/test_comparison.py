import numpy as np
import pandas as pd
import pytest

from src.data import load_daily
from src.model import FEATURES, add_features
from scripts.compare_models import METHODS, compare


def test_one_step_ahead_features_do_not_include_same_day_orders_or_aov():
    assert "orders" not in FEATURES
    assert "aov" not in FEATURES
    daily, _ = load_daily("data/demo/daily_sales.csv")
    left = add_features(daily)
    right_source = daily.copy()
    j = 110
    right_source.loc[j, "revenue"] *= 1000
    right = add_features(right_source)
    # A change on day j must not change any features OF day j.
    for name in FEATURES:
        assert left.loc[left.date.eq(daily.loc[j, "date"]), name].iloc[0] == \
               right.loc[right.date.eq(daily.loc[j, "date"]), name].iloc[0]


def test_three_models_have_identical_holdout_rows():
    daily, _ = load_daily("data/demo/daily_sales.csv")
    val, test, report = compare(daily)
    assert len(val) == 35
    assert len(test) == 60
    assert val.date.max() < test.date.min()
    assert report["selected_method"] in METHODS
    for name in METHODS:
        assert test[name].notna().all()
        assert test[name].ge(0).all()
        assert np.isfinite(report["test_metrics"][name]["wape_pct"])
    selected_by_validation = min(METHODS, key=lambda n: report["validation_metrics"][n]["wape_pct"])
    assert report["selected_method"] == selected_by_validation
    pd.testing.assert_series_equal(test.chosen_forecast, test[report["selected_method"]], check_names=False)


def test_short_series_rejected():
    daily = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=75),
                          "revenue": np.ones(75)})
    with pytest.raises(ValueError):
        compare(daily)
