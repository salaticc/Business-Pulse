import numpy as np
import pandas as pd
from src.data import load_daily
from src.model import add_features, backtest


def test_future_values_cannot_change_earlier_features():
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    a = pd.DataFrame({"date": dates, "revenue": np.arange(120) + 100})
    b = a.copy()
    b.loc[b.index[-1], "revenue"] = 10_000_000
    fa, fb = add_features(a), add_features(b)
    # Predictions at each time are determined by earlier sales, not current target.
    pd.testing.assert_frame_equal(fa.iloc[:-1].drop(columns="revenue"), fb.iloc[:-1].drop(columns="revenue"))
    assert fa.iloc[-1]["lag_1"] == fb.iloc[-1]["lag_1"]


def test_demo_data_and_backtest_exist():
    daily, meta = load_daily("data/demo/daily_sales.csv")
    result, report = backtest(daily)
    assert "SYNTHETIC" in meta["source"]
    assert len(result) == 60
    assert report["selected_predictor"] in ("ML", "weekday_baseline")
    assert set(["alert", "predicted", "shortfall_vs_forecast"]).issubset(result.columns)
    assert result["predicted"].ge(0).all()


def test_daily_csv_rejects_duplicate_dates(tmp_path):
    path = tmp_path / "dup.csv"
    pd.DataFrame({"date": ["2025-01-01", "2025-01-01"], "revenue": [1, 2], "orders": [1, 1]}).to_csv(path, index=False)
    try:
        load_daily(path)
        assert False, "Expected ValueError"
    except ValueError as err:
        assert "ровно одна строка" in str(err)


def test_uci_loader_ignores_cancellations_but_keeps_sales_without_customer_id(tmp_path):
    path = tmp_path / "Online Retail.xlsx"
    raw = pd.DataFrame({
        "InvoiceNo": ["100001", "C100001", "100002", "100003", "100004"],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-02", "2011-01-03"]),
        "Country": ["United Kingdom"] * 5,
        "Quantity": [2, -2, 1, 1, 2],
        "UnitPrice": [10, 10, 20, 0, -3],
        "CustomerID": [pd.NA, pd.NA, 1234, 1234, 1234],
    })
    raw.to_excel(path, index=False)
    daily, meta = load_daily(path)
    assert list(daily["revenue"]) == [20.0, 20.0]
    assert list(daily["orders"]) == [1, 1]
    assert meta["cancellation_lines_excluded"] == 1
