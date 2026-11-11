"""Lesson 02: leakage-safe one-day-ahead backtest; demo or UCI data.

Run from repo root:
    python -m scripts.compare_models --input data/demo/daily_sales.csv

At prediction time (morning of date t), observed revenue up to t-1 is
available, but same-day orders/revenue/actual marketing spend are not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data import load_daily
from src.model import FEATURES, add_features, metrics, new_model


METHODS = ("last_same_weekday", "four_week_median", "gradient_boosting")


def compare(daily: pd.DataFrame, validation_days: int = 35, test_days: int = 60):
    """Returns comparable predictions and validation-selected method.

    Holdout order: training -> validation -> test. ML is refit on
    training+validation after its fixed hyperparameters are selected.
    Baseline/ML features refer only to historical actuals.
    """
    if validation_days < 7 or test_days < 7:
        raise ValueError("validation_days и test_days должны быть >= 7")
    data = add_features(daily)
    if len(data) < validation_days + test_days + 60:
        raise ValueError("Недостаточно исторических дней для train / val / test")

    tr_end = len(data) - validation_days - test_days
    val_end = len(data) - test_days
    train = data.iloc[:tr_end].copy()
    val = data.iloc[tr_end:val_end].copy()
    test = data.iloc[val_end:].copy()
    assert train.date.max() < val.date.min() < test.date.min()

    def baselines(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["last_same_weekday"] = result["lag_7"].clip(lower=0)
        result["four_week_median"] = result["baseline"].clip(lower=0)
        return result

    val = baselines(val)
    val_model = new_model().fit(train[FEATURES], train["revenue"])
    val["gradient_boosting"] = np.maximum(0, val_model.predict(val[FEATURES]))
    scores_val = {name: metrics(val["revenue"], val[name]) for name in METHODS}

    # Choose on validation alone. No tuning using test scores.
    selected = min(METHODS, key=lambda name: scores_val[name]["wape_pct"])

    test = baselines(test)
    refit = new_model().fit(data.iloc[:val_end][FEATURES], data.iloc[:val_end]["revenue"])
    test["gradient_boosting"] = np.maximum(0, refit.predict(test[FEATURES]))
    scores_test = {name: metrics(test["revenue"], test[name]) for name in METHODS}
    test["chosen_forecast"] = test[selected]
    test["error"] = test["revenue"] - test["chosen_forecast"]
    test["absolute_error"] = test["error"].abs()

    report = {
        "warning": "Educational evaluation. Demo dataset is synthetic; test metrics are not production performance.",
        "forecast_horizon": "one-day-ahead, using actual prior-day history for each evaluated day",
        "target": "daily revenue (not profit, not loss)",
        "data_available_at_prediction": ["calendar", "revenue through yesterday", "aggregates of past revenue"],
        "prohibited_future_features": ["today's actual orders", "today's actual average order value", "today's actual advertising spend"],
        "training_dates": [str(train.date.min().date()), str(train.date.max().date())],
        "validation_dates": [str(val.date.min().date()), str(val.date.max().date())],
        "test_dates": [str(test.date.min().date()), str(test.date.max().date())],
        "n_days": {"train": len(train), "validation": len(val), "test": len(test)},
        "selection_metric": "WAPE on validation only",
        "selected_method": selected,
        "validation_metrics": scores_val,
        "test_metrics": scores_test,
        "note": "Features for one-step-ahead test use actual history up to preceding day; not a 60-day forecast from test start.",
        "limitations": [
            "No verified labels for true operational incidents; alert precision/recall cannot be calculated.",
            "Source may contain zeros caused by closures or incomplete data, which need separate checks.",
            "Comparing predictions cannot prove causality or financial loss.",
            "Single chronological split; further rolling-origin cross-validation and drift analysis needed.",
        ],
    }
    return val, test, report


def save_results(val: pd.DataFrame, test: pd.DataFrame, report: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    columns = ["date", "revenue", *METHODS]
    val[columns].to_csv(outdir / "comparison_validation.csv", index=False, encoding="utf-8-sig")
    test[columns + ["chosen_forecast", "error", "absolute_error"]].to_csv(
        outdir / "comparison_test.csv", index=False, encoding="utf-8-sig"
    )
    (outdir / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.date, y=test.revenue, name="Факт", mode="lines"))
    labels = {"last_same_weekday":"Неделю назад", "four_week_median":"Медиана 4 недель", "gradient_boosting":"Gradient Boosting"}
    for method in METHODS:
        fig.add_trace(go.Scatter(x=test.date, y=test[method], name=labels[method], mode="lines"))
    fig.update_layout(
        title="Business Pulse ML | сравнение прогнозов (синтетика или исходный датасет)",
        xaxis_title="Дата", yaxis_title="Выручка (единицы исходных данных)",
        template="plotly_white", hovermode="x unified",
        annotations=[dict(text=f"Метод по validation: {report['selected_method']}", x=0, y=1.12, xref="paper", yref="paper", showarrow=False)],
    )
    fig.write_html(str(outdir / "comparison.html"), include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/demo/daily_sales.csv")
    parser.add_argument("--country", default="United Kingdom")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    daily, source = load_daily(args.input, args.country)
    val, test, report = compare(daily)
    report["source"] = source
    save_results(val, test, report, Path(args.output))
    print("Источник:", source["source"])
    print("Выбран по валидации:", report["selected_method"])
    for name in METHODS:
        print(f"{name:20s} val WAPE={report['validation_metrics'][name]['wape_pct']:6.2f}%   test WAPE={report['test_metrics'][name]['wape_pct']:6.2f}%   test MAE={report['test_metrics'][name]['mae']:.2f}")
    print("Файлы: comparison.html, comparison_test.csv, comparison_report.json")


if __name__ == "__main__":
    main()
