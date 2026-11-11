"""Chronological one-step-ahead backtest. Does not use future sales as features."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    "dow", "month", "day", "is_weekend", "lag_1", "lag_7", "lag_14",
    "lag_21", "lag_28", "mean_7", "mean_28", "std_28",
]


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values("date").copy().reset_index(drop=True)
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["day"] = d["date"].dt.day
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    for lag in (1, 7, 14, 21, 28):
        d[f"lag_{lag}"] = d["revenue"].shift(lag)
    past = d["revenue"].shift(1)
    d["mean_7"] = past.rolling(7).mean()
    d["mean_28"] = past.rolling(28).mean()
    d["std_28"] = past.rolling(28).std().fillna(0)
    d["baseline"] = d[["lag_7", "lag_14", "lag_21", "lag_28"]].median(axis=1)
    return d.dropna(subset=FEATURES + ["baseline"]).reset_index(drop=True)


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "mae": round(float(mean_absolute_error(y, p)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 2),
        "wape_pct": round(float(100 * np.abs(y-p).sum() / max(1, np.abs(y).sum())), 2),
    }


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=150, max_leaf_nodes=15, min_samples_leaf=12,
        learning_rate=0.055, l2_regularization=10.0,
        random_state=42,
    )


def backtest(daily: pd.DataFrame, test_days: int = 60, val_days: int = 35):
    d = add_features(daily)
    if len(d) < test_days + val_days + 60:
        raise ValueError("Слишком короткий ряд. Нужно минимум 155 дней ПОСЛЕ лагов (28 дней).")
    train = d.iloc[: -(test_days + val_days)].copy()
    val = d.iloc[-(test_days + val_days):-test_days].copy()
    test = d.iloc[-test_days:].copy()

    # Model selection uses VALIDATION only, then evaluates once on TEST.
    ml_val = new_model().fit(train[FEATURES], train["revenue"])
    val["ml_pred"] = np.clip(ml_val.predict(val[FEATURES]), 0, None)
    val["baseline_pred"] = val["baseline"].clip(lower=0)
    val_ml = metrics(val["revenue"], val["ml_pred"])
    val_base = metrics(val["revenue"], val["baseline_pred"])
    selected = "ML" if val_ml["wape_pct"] < val_base["wape_pct"] else "weekday_baseline"
    selected_val = val["ml_pred"] if selected == "ML" else val["baseline_pred"]
    # One-sided residual calibration from past validation (observed revenue below prediction).
    shortfalls = np.maximum(selected_val.to_numpy() - val["revenue"].to_numpy(), 0)
    threshold = float(np.quantile(shortfalls, 0.90))

    ml_test = new_model().fit(pd.concat([train, val])[FEATURES], pd.concat([train, val])["revenue"])
    test["ml_pred"] = np.clip(ml_test.predict(test[FEATURES]), 0, None)
    test["baseline_pred"] = test["baseline"].clip(lower=0)
    test["predicted"] = test["ml_pred"] if selected == "ML" else test["baseline_pred"]
    test["shortfall_vs_forecast"] = (test["predicted"] - test["revenue"]).clip(lower=0)
    test["drop_pct"] = np.where(
        test["predicted"] > 0,
        100 * test["shortfall_vs_forecast"] / test["predicted"].clip(lower=1),
        0.0,
    )
    test["alert"] = (test["shortfall_vs_forecast"] > threshold) & (test["drop_pct"] >= 25)
    test["alert_reason"] = np.where(
        test["alert"], "Проверьте полноту загрузки данных, график работы и число заказов", ""
    )
    report = {
        "evaluation": "one-step-ahead rolling historical predictions; daily lag features use actual past observations",
        "selection_basis": "validation WAPE only",
        "selected_predictor": selected,
        "train_dates": [str(train["date"].min().date()), str(train["date"].max().date())],
        "validation_dates": [str(val["date"].min().date()), str(val["date"].max().date())],
        "test_dates": [str(test["date"].min().date()), str(test["date"].max().date())],
        "validation": {"ml": val_ml, "weekday_baseline": val_base},
        "test": {
            "ml": metrics(test["revenue"], test["ml_pred"]),
            "weekday_baseline": metrics(test["revenue"], test["baseline_pred"]),
        },
        "alert_calibration": {
            "min_relative_drop_pct": 25,
            "absolute_shortfall_threshold_from_validation": round(threshold, 2),
            "calibration_method": "90th percentile of one-sided shortfalls on validation; heuristic, not a statistical guarantee",
            "alert_days_test": int(test["alert"].sum()),
        },
    }
    return test, report
