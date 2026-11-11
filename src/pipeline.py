"""CLI: python -m src.pipeline --input data/demo/daily_sales.csv"""
import argparse
import json
from pathlib import Path
from src.data import load_daily
from src.model import backtest
from src.visualize import save_html


def main():
    p = argparse.ArgumentParser(description="G-Analytics Business Pulse ML — backtest")
    p.add_argument("--input", default="data/demo/daily_sales.csv")
    p.add_argument("--country", default="United Kingdom")
    p.add_argument("--output", default="outputs")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    daily, source_meta = load_daily(args.input, args.country)
    test, report = backtest(daily)
    report["source"] = source_meta
    report["caveats"] = [
        "Прогноз ожидаемых продаж не доказывает потери дохода или причинное влияние события.",
        "Выручка UCI — сумма положительных товарных строк, без возвратов и себестоимости.",
        "Нули могут означать выходной, сбой выгрузки или отсутствие транзакций.",
        "Датасет UCI 2010–2011 гг. и не представляет российский рынок 2026 года.",
        "Алерты не имеют исторических меток инцидентов: precision/recall пока неизвестны.",
    ]
    test.to_csv(out / "backtest.csv", index=False, encoding="utf-8-sig")
    test.loc[test["alert"], ["date", "revenue", "predicted", "shortfall_vs_forecast", "drop_pct", "orders", "alert_reason"]].to_csv(
        out / "alerts.csv", index=False, encoding="utf-8-sig")
    with open(out / "report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    save_html(test, report, out / "dashboard.html")
    print("Источник:", source_meta["source"])
    print("Выбранный подход (по валидации):", report["selected_predictor"])
    print("TEST WAPE baseline:", report["test"]["weekday_baseline"]["wape_pct"], "%")
    print("TEST WAPE ML:", report["test"]["ml"]["wape_pct"], "%")
    print("Дней с предупреждениями:", report["alert_calibration"]["alert_days_test"])
    print("Отчёт:", out / "dashboard.html")
    print("Разбор причин и бизнес-решения — ВАШ следующий этап, см. TALENT_TEST.md")


if __name__ == "__main__":
    main()
