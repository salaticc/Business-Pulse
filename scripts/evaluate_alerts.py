"""v0.6 local experiments. No send flag, notification import or network client."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from hashlib import sha256
from html import escape
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pandas as pd

from src.alerts import AlertMonitor, AlertPolicy, Observation, _amount
from src.alert_evaluation import IncidentLabel, score_alerts, select_policy

ROOT = Path(__file__).resolve().parents[1]


def boolean(value) -> bool:
    if str(value).lower() not in {"true", "false"}:
        raise ValueError("Значения полноты/синтетического происхождения должны быть True или False")
    return str(value).lower() == "true"


def observations_from_frame(frame: pd.DataFrame) -> list[Observation]:
    result = []
    for row in frame.to_dict("records"):
        record = {f.name: row[f.name] for f in fields(Observation)}
        for name in ("forecast", "actual"):
            record[name] = float(record[name]) if record[name] not in ("", None) else None
        for name in ("history_complete", "actual_complete"):
            record[name] = boolean(record[name])
        for name in ("forecast_at", "history_closed_at", "history_complete_at", "history_through",
                     "actual_complete_at", "data_through", "source_updated_at"):
            value = record[name]
            record[name] = None if value is None or pd.isna(value) or value == "" else value
        result.append(Observation(**record))
    return result


def labels_from_frame(frame: pd.DataFrame, severity_costs: dict) -> list[IncidentLabel]:
    if not severity_costs or any(not _amount(c) for c in severity_costs.values()):
        raise ValueError("Нужны конечные неотрицательные стоимости тяжести")
    result = []
    for row in frame.to_dict("records"):
        identifier = row["incident_id"] or None
        severity = row["severity"] or None
        multiplier = float(row["cost_multiplier"])
        if not _amount(multiplier) or (not identifier and multiplier != 0):
            raise ValueError("Некорректный множитель стоимости инцидента")
        cost = severity_costs[severity] * multiplier if identifier else 0.
        result.append(IncidentLabel(row["stream_id"], row["date"], identifier, severity, cost, boolean(row["synthetic"])))
    return result


def policies_from_config(config: dict) -> list[AlertPolicy]:
    common = {key: config[key] for key in ("cooldown_days", "max_data_age_hours", "high_drop_pct", "critical_drop_pct")}
    return [AlertPolicy(relative, absolute, **common)
            for relative in config["relative_drop_pct"] for absolute in config["absolute_shortfall"]]


def check_partitions(frame: pd.DataFrame) -> None:
    if not set(frame["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Неизвестный split")
    for name in ("validation", "test"):
        if frame.loc[frame.split.eq(name)].empty:
            raise ValueError(f"Не задан {name}")
    if frame.loc[frame.split.eq("validation"), "date"].max() >= frame.loc[frame.split.eq("test"), "date"].min():
        raise ValueError("Validation должен завершаться раньше TEST")
    train = frame.loc[frame.split.eq("train")]
    if not train.empty and train.date.max() >= frame.loc[frame.split.eq("validation"), "date"].min():
        raise ValueError("Train должен завершаться раньше validation")
    if frame.duplicated(["stream_id", "date"]).any():
        raise ValueError("Повтор stream/date")
    for _, group in frame.groupby("stream_id"):
        dates = pd.to_datetime(group.date.sort_values())
        if not dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError("Нужна строка на каждый календарный день, даже если actual отсутствует")


def experiment(observations: pd.DataFrame, labels: pd.DataFrame, config: dict, *, validation_only: bool = False) -> tuple[dict, list[dict], list[dict], dict]:
    check_partitions(observations)
    val = observations.loc[observations.split.eq("validation")]
    val_labels = labels.loc[labels.split.eq("validation")]
    policies = policies_from_config(config)
    selected = select_policy(observations_from_frame(val), labels_from_frame(val_labels, config["miss_cost_by_severity"]),
                             policies, config["false_alert_cost"])
    # Test labels / forecasts are first interpreted only AFTER policy/state freeze.
    report = {
        "version": "0.6", "source": "SYNTHETIC educational incidents, not real business ground truth",
        "config": config, "selected_policy": asdict(selected.policy),
        "selection": "validation total error cost; tie: fewer false warnings, then configured policy order",
        "validation_dates": [val.date.min(), val.date.max()],
        "test_dates": [observations.loc[observations.split.eq('test'), 'date'].min(),
                       observations.loc[observations.split.eq('test'), 'date'].max()],
        "validation_metrics": selected.metrics, "threshold_comparison": selected.comparison,
        "test_evaluated": not validation_only,
        "units": {"confusion": "one scheduled stream/calendar-day, after suppression; blocked = no business warning",
                  "candidate_confusion": "before suppression, after quality gate",
                  "miss_cost": "once per entirely undetected labelled incident, including blocked days",
                  "false_cost": "each emitted business warning on a negative labelled day",
                  "quality_warnings": "reported separately; not scored as false business warnings"},
        "limitations": [
            "All data, availability timestamps, causes, severity and costs are simulated.",
            "Synthetic same-generator TEST checks this implementation, not real commercial effectiveness.",
            "No inference of actual financial loss, prevented losses or causal intervention effects.",
            "Late detection within an incident counts as detected; delay is reported, not penalized.",
            "True-warning review cost and quality-warning costs are excluded; no incident-level TN is defined.",
            "No external delivery; local sequential state is not a concurrent production service.",
        ],
    }
    journal = [{"split": "validation", **r} for r in selected.journal]
    ledger = [{"split": "validation", **r} for r in selected.incidents]
    monitor = AlertMonitor(selected.policy, selected.state)
    if not validation_only:
        test = observations.loc[observations.split.eq("test")]
        test_labels = labels.loc[labels.split.eq("test")]
        val_ids = {(x.stream_id, x.incident_id) for x in labels_from_frame(val_labels, config["miss_cost_by_severity"]) if x.incident_id}
        parsed_test_labels = labels_from_frame(test_labels, config["miss_cost_by_severity"])
        if val_ids & {(x.stream_id, x.incident_id) for x in parsed_test_labels if x.incident_id}:
            raise ValueError("Один размеченный инцидент не должен пересекать validation/TEST")
        metrics, test_ledger, test_journal = score_alerts(
            monitor.run(observations_from_frame(test)),
            parsed_test_labels, config["false_alert_cost"],
        )
        report["test_metrics"] = metrics
        journal.extend({"split": "test", **r} for r in test_journal)
        ledger.extend({"split": "test", **r} for r in test_ledger)
    return report, journal, ledger, monitor.state


def save_results(output: Path, report: dict, journal: list[dict], ledger: list[dict], state: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "alert_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "incident_journal.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in journal), encoding="utf-8")
    flat = [{**r, "reasons": "; ".join(r["reasons"]), "evidence": json.dumps(r["evidence"], ensure_ascii=False, sort_keys=True)} for r in journal]
    pd.DataFrame(flat).to_csv(output / "incident_journal.csv", index=False, lineterminator="\n")
    pd.DataFrame(ledger).to_csv(output / "incident_outcomes.csv", index=False, lineterminator="\n")
    comparison = pd.DataFrame(report["threshold_comparison"])
    comparison.to_csv(output / "threshold_comparison.csv", index=False, lineterminator="\n")
    (output / "monitor_state.json").write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    parts = ["<!doctype html><html lang='ru'><meta charset='utf-8'><title>Business Pulse v0.6</title>",
             "<style>body{font:16px system-ui;margin:32px;color:#17324d}table{border-collapse:collapse;font-size:13px}"
             "td,th{border:1px solid #ccd7e3;padding:7px}th{background:#eef4fa}.scroll{overflow-x:auto}pre{white-space:pre-wrap}</style>",
             "<h1>Business Pulse v0.6 · Alert Intelligence</h1><p><strong>Только синтетическая учебная разметка.</strong> "
             "Условные стоимости не обозначают реальные убытки. Внешних отправок нет.</p>",
             "<h2>Выбор только по validation</h2><pre>" + escape(json.dumps(report["selected_policy"], ensure_ascii=False, indent=2)) + "</pre>",
             "<p>Сначала минимум стоимости, затем число ложных сообщений, затем порядок сетки. "
             "Дневные TP/FP/FN/TN ниже относятся к выданным локальным предупреждениям после подавления. "
             "Стоимость пропуска начисляется один раз за целый невыявленный инцидент.</p>",
             "<div class='scroll'>" + comparison.to_html(index=False, na_rep="undefined", float_format=lambda v: f"{v:.4f}") + "</div>"]
    for name in ("validation", "test"):
        if f"{name}_metrics" in report:
            parts += [f"<h2>{name.upper()}: только выбранная политика</h2><pre>",
                      escape(json.dumps(report[f"{name}_metrics"], ensure_ascii=False, indent=2)), "</pre>"]
    parts += ["<h2>Размеченные инциденты</h2><div class='scroll'>",
              pd.DataFrame(ledger).to_html(index=False, na_rep="undefined"), "</div>",
              "<h2>Журнал решений и подтверждение учебной истиной</h2><div class='scroll'>",
              pd.DataFrame(flat).drop(columns=["evidence"]).to_html(index=False, na_rep="undefined"), "</div>",
              "<p>Полные использованные времена, значения и snapshot_id — в evidence каждого события JSONL/CSV. "
              "Разметка присоединяется после решения и не участвует в detector.</p><h2>Ограничения</h2><ul>"]
    parts.extend("<li>" + escape(line) + "</li>" for line in report["limitations"])
    parts.append("</ul></html>")
    (output / "alert_report.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Business Pulse v0.6: local synthetic alert evaluation")
    parser.add_argument("--observations", default="data/synthetic_incidents/observations.csv")
    parser.add_argument("--labels", default="data/synthetic_incidents/labels.csv")
    parser.add_argument("--config", default="configs/alert_v06.json")
    parser.add_argument("--output", default="outputs/v06")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        observations = pd.read_csv(args.observations, keep_default_na=False, dtype={"date": str})
        labels = pd.read_csv(args.labels, keep_default_na=False, dtype={"date": str})
        report, journal, ledger, state = experiment(observations, labels, config, validation_only=args.validation_only)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    report["reproducibility"] = {
        "python": platform.python_version(), "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn")},
        "input_sha256": {name: sha256(Path(path).read_bytes()).hexdigest() for name, path in (
            ("observations", args.observations), ("labels", args.labels), ("config", args.config))},
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
                        for p in sorted([*ROOT.glob('src/*.py'), *ROOT.glob('scripts/*.py')])},
    }
    save_results(Path(args.output), report, journal, ledger, state)
    print(report["source"])
    print("Selected on validation:", report["selected_policy"])
    for name in ("validation", "test"):
        if f"{name}_metrics" in report:
            print(name, json.dumps(report[f"{name}_metrics"], ensure_ascii=False))
    print("Report:", Path(args.output) / "alert_report.html")


if __name__ == "__main__":
    main()
