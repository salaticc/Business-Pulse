import numpy as np
import pandas as pd
import pytest

from src.data import load_daily
from src.model import add_features


@pytest.mark.parametrize("column,value", [
    ("revenue", -1), ("revenue", np.inf), ("revenue", np.nan),
    ("orders", -1), ("orders", np.inf), ("orders", np.nan), ("orders", 1.5),
    ("date", None), ("date", "invalid-date"),
])
def test_csv_invalid_values_rejected(tmp_path, column, value):
    data = {"date": ["2025-01-01"], "revenue": [10.], "orders": [1.]}
    data[column] = [value]
    path = tmp_path / "invalid.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_daily(path)


def test_missing_dates_not_silently_zeroed_and_custom_csv_not_mislabeled(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("date,revenue,orders\n2025-01-01,12,1\n2025-01-03,0,0\n")
    with pytest.raises(ValueError, match="Пропущены"):
        load_daily(path)
    daily, meta = load_daily(path, missing_days="zero")
    assert daily.revenue.tolist() == [12., 0., 0.]
    assert meta["assumed_zero_dates"] == ["2025-01-02"]
    assert meta["source_kind"] == "unverified"
    assert not meta["completeness_verified"]


@pytest.mark.parametrize("case", ["gap", "duplicate", "nat", "negative", "infinite", "intraday"])
def test_feature_builder_validates_direct_inputs(case):
    daily = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=100), "revenue": np.ones(100)})
    if case == "gap":
        daily = daily.drop(index=50)
    elif case == "duplicate":
        daily.loc[50, "date"] = daily.loc[49, "date"]
    elif case == "nat":
        daily.loc[50, "date"] = pd.NaT
    elif case == "intraday":
        daily.loc[50, "date"] += pd.Timedelta(hours=1)
    else:
        daily.loc[50, "revenue"] = -1 if case == "negative" else np.inf
    with pytest.raises(ValueError):
        add_features(daily)


def test_uci_quality_audit_and_unknown_days(tmp_path):
    path = tmp_path / "retail.xlsx"
    raw = pd.DataFrame({
        "InvoiceNo": ["1", "1", "C2", "3", "4", "5"],
        "InvoiceDate": ["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-03", "2011-01-03", "bad"],
        "Country": ["United Kingdom"] * 6,
        "Quantity": [1, 1, -1, 2, 1, 1], "UnitPrice": [10, 10, 10, 10, np.inf, 10],
    })
    raw.to_excel(path, index=False)
    with pytest.raises(ValueError, match="Пропущены"):
        load_daily(path)
    daily, meta = load_daily(path, missing_days="zero")
    assert daily.revenue.tolist() == [20., 0., 20.]
    assert meta["cancellation_lines_excluded"] == 1
    assert meta["invalid_non_cancellation_lines_excluded"] == 2
    assert meta["exact_duplicate_rows_detected"] == 1
    assert meta["assumed_zero_dates"] == ["2011-01-02"]


def test_unsorted_input_is_ordered_before_lags():
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=100), "revenue": np.arange(100.)})
    pd.testing.assert_frame_equal(add_features(frame), add_features(frame.iloc[::-1]))
