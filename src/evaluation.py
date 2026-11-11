"""Expanding-origin evaluation of one-day forecasts with an isolated final holdout.

Forecast date t after the prior trading day closes and ingestion is confirmed,
before t opens. Calendar-day actuals through t-1 are assumed complete here;
the offline evaluator does not implement or verify the operational gate.
Models are refit at each validation origin, then once before the final test.
Within each window weights are frozen; lag features update from observed history.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
import logging
import platform

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data import validate_daily
from src.model import FEATURES, add_features, metrics, new_model

LOGGER = logging.getLogger(__name__)
METHODS = ("last_same_weekday", "four_week_median", "gradient_boosting")
METRICS = ("mae", "rmse", "wape_pct")


@dataclass(frozen=True)
class EvaluationConfig:
    n_splits: int = 3
    validation_days: int = 28
    test_days: int = 60
    min_train_days: int = 60

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} должен быть положительным целым числом")
        if self.n_splits < 2:
            raise ValueError("Нужно минимум два временных окна")
        if self.validation_days < 7 or self.test_days < 7:
            raise ValueError("validation_days и test_days должны быть >= 7")
        if self.min_train_days < 60:
            raise ValueError("min_train_days должен быть >= 60 после прогрева лагов")


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: int  # exclusive; train always starts at 0
    validation_end: int  # exclusive


def expanding_folds(n_development: int, config: EvaluationConfig) -> tuple[Fold, ...]:
    """Positions refer to feature-ready development rows; no test rows accepted."""
    first_origin = n_development - config.n_splits * config.validation_days
    if first_origin < config.min_train_days:
        required = 28 + config.min_train_days + config.n_splits * config.validation_days + config.test_days
        raise ValueError(f"Недостаточно истории: нужно минимум {required} календарных дней")
    return tuple(
        Fold(i + 1, first_origin + i * config.validation_days,
             first_origin + (i + 1) * config.validation_days)
        for i in range(config.n_splits)
    )


def _dates(frame: pd.DataFrame) -> list[str]:
    return [str(frame.date.iloc[0].date()), str(frame.date.iloc[-1].date())]


def _predict(train: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    if train.date.max() >= window.date.min():
        raise ValueError("Обучение должно завершиться до начала прогнозируемого окна")
    result = window[["date", "revenue"]].copy()
    result["last_same_weekday"] = window["lag_7"].to_numpy()
    result["four_week_median"] = window["baseline"].to_numpy()
    # One native thread gives stable reductions and avoids tiny-data oversubscription.
    with threadpool_limits(limits=1):
        model = new_model().fit(train[FEATURES], train["revenue"])
        result["gradient_boosting"] = np.maximum(0, model.predict(window[FEATURES]))
    return result


def _scores(predictions: pd.DataFrame) -> dict:
    return {name: metrics(predictions.revenue, predictions[name]) for name in METHODS}


def select_on_history(
    history: pd.DataFrame, config: EvaluationConfig,
) -> tuple[pd.DataFrame, dict]:
    """Only development data enter this function; test targets are inaccessible.

    Choose by pooled out-of-fold WAPE, without rounding. With identical target
    dates this ranks methods identically to total absolute error / pooled MAE.
    Undefined fold WAPE is null, not zero; MAE/RMSE remain defined.
    """
    data = add_features(history)
    folds = expanding_folds(len(data), config)
    predictions, fold_reports = [], []
    for fold in folds:
        train = data.iloc[:fold.train_end]
        window = data.iloc[fold.train_end:fold.validation_end]
        pred = _predict(train, window)
        pred.insert(0, "fold", fold.number)
        predictions.append(pred)
        fold_reports.append({
            "fold": fold.number,
            "train_dates": _dates(train), "validation_dates": _dates(window),
            "n_train": len(train), "n_validation": len(window),
            "model_fit_through": str(train.date.max().date()),
            "metrics": _scores(pred),
        })
        LOGGER.info("Fold %s: train through %s, validate %s", fold.number, train.date.max().date(), _dates(window))
    oof = pd.concat(predictions, ignore_index=True)
    pooled = _scores(oof)
    if pooled[METHODS[0]]["wape_pct"] is None:
        raise ValueError("WAPE не определён: суммарная выручка всех CV-окон равна нулю")
    selected = min(METHODS, key=lambda name: pooled[name]["wape_pct"])
    dispersion = {}
    for name in METHODS:
        dispersion[name] = {}
        for metric in METRICS:
            values = [f["metrics"][name][metric] for f in fold_reports]
            valid = [v for v in values if v is not None]
            dispersion[name][metric] = {
                "mean": float(np.mean(valid)) if valid else None,
                "std_population": float(np.std(valid, ddof=0)) if valid else None,
                "min": float(min(valid)) if valid else None,
                "max": float(max(valid)) if valid else None,
                "defined_windows": len(valid),
            }
    report = {
        "selected_method": selected,
        "selection_metric": "pooled out-of-fold WAPE, full precision; no test access",
        "tie_break_order": list(METHODS),
        "development_dates": _dates(history),
        "folds": fold_reports, "cv_pooled_metrics": pooled,
        "cv_window_dispersion": dispersion,
    }
    return oof, report


def evaluate(
    daily: pd.DataFrame, config: EvaluationConfig | None = None, *,
    evaluate_test: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Freeze selection before forming or scoring final-test feature rows."""
    config = config or EvaluationConfig()
    validate_daily(daily)
    ordered = daily.sort_values("date").reset_index(drop=True)
    if len(ordered) <= config.test_days:
        raise ValueError("Недостаточно дней до финального теста")
    history = ordered.iloc[:-config.test_days].copy()
    cv, selection = select_on_history(history, config)
    selected = selection["selected_method"]  # fixed before any test features / metrics
    report = {
        "version": "0.5",
        "config": asdict(config),
        "forecast_contract": {
            "revision": 2,
            "horizon_days": 1,
            "prediction_time": "after prior trading day closure and confirmed complete ingestion, before target day t opens",
            "lag_basis": "calendar days; non-trading dates require confirmed calendar status",
            "operational_gate_implemented": False,
            "historical_availability_verified": False,
            "backtest_assumption": "complete calendar-day actuals through t-1; no as-of ingestion timestamps or historical source versions",
            "features": list(FEATURES),
            "actuals_available_through": "t-1",
            "fit_policy": "once per expanding CV origin; once on all development before final test; frozen within windows",
            "warmup_days": 28,
            "native_threads": 1,
        },
        **selection,
        "test_dates": _dates(ordered.iloc[-config.test_days:]),
        "test_evaluated": evaluate_test,
        "model_parameters": new_model().get_params(),
        "reproducibility": {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in (
                "numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl", "plotly", "openpyxl",
            )},
            "development_sha256": sha256(history[["date", "revenue"]].to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        },
        "limitations": [
            "Bundled demo is synthetic; raw UCI transactions are not bundled. Conclusions apply only to the supplied source.",
            "If using the bundled demo: v0.4 final-test results were already published; this is a code-isolated holdout, not a previously unseen experiment.",
            "Only a few correlated time windows; dispersion is descriptive, not a confidence interval or proof of significance.",
            "Daily source completeness / opening calendar are not inferable from sales alone.",
            "No incident labels: forecast errors do not establish alert precision/recall, causes, financial loss or profit.",
        ],
    }
    test = None
    if evaluate_test:
        data = add_features(ordered)
        train = data.loc[data.date.lt(history.date.max() + pd.Timedelta(days=1))]
        window = data.loc[data.date.gt(history.date.max())]
        if len(window) != config.test_days:
            raise ValueError("Нарушена граница финального теста")
        test = _predict(train, window)
        test["chosen_forecast"] = test[selected]
        test["error"] = test.revenue - test.chosen_forecast
        test["absolute_error"] = test.error.abs()
        report["final_fit_dates"] = _dates(train)
        report["final_fit_rows"] = len(train)
        report["test_metrics"] = _scores(test)
        report["selected_test_metrics"] = report["test_metrics"][selected]
    return cv, test, report
