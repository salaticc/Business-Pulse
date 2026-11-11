"""Business Pulse v0.5: reproducible walk-forward evaluation CLI."""
from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.data import load_daily
from src.evaluation import EvaluationConfig, METHODS, evaluate

ROOT = Path(__file__).resolve().parents[1]


def save_results(cv: pd.DataFrame, test: pd.DataFrame | None, report: dict, outdir: Path) -> None:
    """Stable JSON/CSV/HTML; no timestamps or random plot IDs."""
    outdir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(outdir / "cv_predictions.csv", index=False, lineterminator="\n")
    rows = []
    for fold in report["folds"]:
        for name in METHODS:
            rows.append({
                "fold": fold["fold"], "method": name,
                "train_start": fold["train_dates"][0], "train_end": fold["train_dates"][1],
                "validation_start": fold["validation_dates"][0], "validation_end": fold["validation_dates"][1],
                "n_train": fold["n_train"], "n_validation": fold["n_validation"],
                **fold["metrics"][name],
            })
    fold_table = pd.DataFrame(rows)
    fold_table.to_csv(outdir / "cv_metrics.csv", index=False, lineterminator="\n")
    if test is not None:
        test.to_csv(outdir / "final_test_predictions.csv", index=False, lineterminator="\n")
    else:
        # Avoid confusing a prior full run's test file with a validation-only run.
        (outdir / "final_test_predictions.csv").unlink(missing_ok=True)
    (outdir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    summary = pd.DataFrame(report["cv_pooled_metrics"]).T
    summary.index.name = "method"
    parts = [
        "<!doctype html><html lang='ru'><meta charset='utf-8'><title>Business Pulse v0.5</title>",
        "<style>body{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 20px;color:#172b4d}"
        "table{border-collapse:collapse;margin:20px 0;width:100%}th,td{padding:9px;border:1px solid #dce3eb;text-align:right}"
        "th{background:#edf3fa}h1,h2{color:#123b61}</style>",
        "<h1>G-Analytics Business Pulse · v0.5</h1>",
        f"<p>Источник: {escape(report.get('source', {}).get('source', 'DataFrame: provenance unverified'))}</p>",
        f"<p>Полнота выгрузки не подтверждена. Дат, явно заполненных нулями: "
        f"{len(report.get('source', {}).get('assumed_zero_dates', []))}.</p>",
        "<p>Контракт: прогноз после закрытия предыдущего торгового дня и подтверждения полноты загрузки, "
        "до открытия прогнозируемого дня. В историческом расчёте готовность данных предполагается; "
        "проверка закрытия и фактического времени доступности не реализована.</p>",
        f"<p>Выбран по pooled CV WAPE: <strong>{escape(report['selected_method'])}</strong>. "
        "Прогноз на один день; внутри окон доступны факты до вчера, веса ML фиксированы.</p>",
        "<h2>Сравнение на временной кросс-валидации</h2>", summary.to_html(float_format=lambda v: f"{v:.4f}"),
        "<h2>Каждое окно</h2>", fold_table.to_html(index=False, float_format=lambda v: f"{v:.4f}"),
    ]
    spread = []
    for method, scores in report["cv_window_dispersion"].items():
        for metric, values in scores.items():
            spread.append({"method": method, "metric": metric, **values})
    parts += ["<h2>Разброс по окнам</h2><p>std_population: ddof=0; это описательный разброс, не доверительный интервал.</p>",
              pd.DataFrame(spread).to_html(index=False, float_format=lambda v: f"{v:.4f}")]
    if test is not None:
        parts += ["<h2>Финальный тест после фиксации выбора</h2>",
                  pd.DataFrame(report["test_metrics"]).T.to_html(float_format=lambda v: f"{v:.4f}")]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.date, y=test.revenue, name="Факт"))
        for name in METHODS:
            fig.add_trace(go.Scatter(x=test.date, y=test[name], name=name))
        fig.update_layout(template="plotly_white", hovermode="x unified", yaxis_title="Выручка, единицы источника")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=True, div_id="final-test-chart"))
    else:
        parts.append("<p>Финальный тест зарезервирован и не оценивался.</p>")
    parts.append("<h2>Ограничения</h2><ul>")
    parts.extend(f"<li>{escape(line)}</li>" for line in report["limitations"])
    parts.append("</ul><p>Полная конфигурация, параметры и контрольные суммы — в evaluation_report.json.</p></html>")
    (outdir / "evaluation.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Business Pulse v0.5 — rolling-origin ML Evaluation")
    parser.add_argument("--input", default="data/demo/daily_sales.csv")
    parser.add_argument("--country", default="United Kingdom")
    parser.add_argument("--output", default="outputs/v05")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-days", type=int, default=28)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--missing-days", choices=("error", "zero"), default="error")
    parser.add_argument("--validation-only", action="store_true", help="Reserve test dates but do not predict or score them")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = EvaluationConfig(args.folds, args.validation_days, args.test_days, args.min_train_days)
        daily, source = load_daily(args.input, args.country, missing_days=args.missing_days)
        cv, test, report = evaluate(daily, config, evaluate_test=not args.validation_only)
    except ValueError as error:
        parser.error(str(error))
    report["source"] = source
    report["reproducibility"]["input_sha256"] = sha256(Path(args.input).read_bytes()).hexdigest()
    code_files = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"),
                         ROOT / "requirements.txt", ROOT / "requirements-lock.txt"])
    report["reproducibility"]["code_sha256"] = {
        path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in code_files if path.exists()
    }
    save_results(cv, test, report, Path(args.output))
    print("Источник:", source["source"])
    print("Выбран по CV:", report["selected_method"])
    for name in METHODS:
        print(f"{name}: CV WAPE={report['cv_pooled_metrics'][name]['wape_pct']:.4f}%")
    print("Финальный тест:", "оценён после фиксации выбора" if test is not None else "не оценивался")
    print("Отчёт:", Path(args.output) / "evaluation.html")


if __name__ == "__main__":
    main()
