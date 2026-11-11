from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_alerts import experiment, labels_from_frame, observations_from_frame, save_results
from src.alerts import AlertMonitor, AlertPolicy
from src.synthetic_incidents import SyntheticConfig, generate, save_dataset


@pytest.fixture
def lab():
    config = json.loads(Path("configs/alert_v06.json").read_text(encoding="utf-8"))
    obs, labels, daily, manifest = generate()
    return obs, labels, daily, manifest, config


def test_synthetic_labels_and_forecasts_are_reproducible_and_not_cost_defined(tmp_path):
    a = generate()
    b = generate(miss_cost_by_severity={"mild": 1, "major": 2, "critical": 3})
    for i in range(3):
        pd.testing.assert_frame_equal(a[i], b[i])
    x, y = tmp_path / "a", tmp_path / "b"
    save_dataset(x)
    save_dataset(y)
    for p in x.iterdir():
        assert p.read_bytes() == (y / p.name).read_bytes()
    assert a[1].synthetic.all()
    assert a[1].incident_id.ne("").any() and a[1].incident_id.eq("").any()
    assert "SYNTHETIC" in a[3]["source"]


def test_forecasts_only_exist_when_all_used_history_was_available(lab):
    obs, _, daily, _, _ = lab
    by_date = daily.set_index("date")
    for o in observations_from_frame(obs):
        day = pd.Timestamp(o.date)
        history = by_date.loc[(by_date.index >= day-pd.Timedelta(days=28)) & (by_date.index < day)]
        available = max(datetime.fromisoformat(s) for s in history.available_at) <= datetime.fromisoformat(o.forecast_at)
        assert available == o.history_complete
        if available:
            expected = np.median([by_date.loc[day-pd.Timedelta(days=lag), "revenue"] for lag in (7, 14, 21, 28)])
            assert o.forecast == pytest.approx(expected)
        else:
            assert o.forecast is None or np.isnan(o.forecast)


def test_more_future_days_do_not_change_validation_generation():
    a, a_truth, _, _ = generate()
    b, b_truth, _, _ = generate(SyntheticConfig(test_days=100))
    pd.testing.assert_frame_equal(a[a.split.ne("test")], b[b.split.ne("test")])
    pd.testing.assert_frame_equal(a_truth[a_truth.split.ne("test")], b_truth[b_truth.split.ne("test")])


def test_unfinished_ingestion_does_not_expose_future_completion_times(lab):
    for row in lab[0].to_dict("records"):
        if not row["history_complete"]:
            assert pd.isna(row["history_complete_at"])
        if not row["actual_complete"]:
            assert pd.isna(row["actual_complete_at"])
        if row["synthetic_quality_scenario"] != "future_source_time":
            assert datetime.fromisoformat(row["source_updated_at"]) <= datetime.fromisoformat(row["snapshot_at"])


def test_final_values_labels_and_costs_cannot_change_selection(lab):
    obs, labels, _, _, config = lab
    a, _, _, _ = experiment(obs, labels, config)
    changed = obs.copy()
    changed.loc[changed.split.eq("test"), "actual"] = 0.
    changed_labels = labels.copy()
    mask = changed_labels.split.eq("test") & changed_labels.incident_id.ne("")
    changed_labels.loc[mask, "cost_multiplier"] *= 1000
    b, _, _, _ = experiment(changed, changed_labels, config)
    for field in ("selected_policy", "threshold_comparison", "validation_metrics"):
        assert a[field] == b[field]
    assert a["test_metrics"] != b["test_metrics"]
    # Validation-only must not even parse invalid test-only observations/labels.
    changed.loc[changed.split.eq("test"), "forecast_at"] = "invalid-test-timestamp"
    changed_labels.loc[changed_labels.split.eq("test"), "severity"] = "unknown-test-severity"
    c, journal, _, _ = experiment(changed, changed_labels, config, validation_only=True)
    assert c["selected_policy"] == a["selected_policy"]
    assert not c["test_evaluated"] and "test_metrics" not in c
    assert {r["split"] for r in journal} == {"validation"}


def test_selected_policy_minimizes_validation_cost_and_only_it_is_tested(lab):
    obs, labels, _, _, config = lab
    report, journal, ledger, state = experiment(obs, labels, config)
    table = report["threshold_comparison"]
    best = min(table, key=lambda r: (r["total_error_cost"], r["false_business_warnings"], r["policy_index"]))
    assert report["selected_policy"]["relative_drop_pct"] == best["relative_threshold_pct"]
    assert report["selected_policy"]["absolute_shortfall"] == best["absolute_threshold"]
    assert len(table) == 12
    for row in table:
        assert row["total_error_cost"] == row["false_alert_cost"] + row["missed_incident_cost"]
        assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == 90
    assert len(journal) == 180
    assert {r["split"] for r in ledger} == {"validation", "test"}
    # Production-like carryover: state after validation is reused, not reset at TEST.
    _, _, _, validation_state = experiment(obs, labels, config, validation_only=True)
    selected = AlertPolicy(**report["selected_policy"])
    manual = AlertMonitor(selected, validation_state)
    manual_test = manual.run(observations_from_frame(obs.loc[obs.split.eq("test")]))
    assert [r["emitted"] for r in manual_test] == [r["emitted"] for r in journal if r["split"] == "test"]
    assert manual.state == state


def test_no_network_during_alert_experiment(monkeypatch, lab):
    def forbidden(*args, **kwargs):
        raise AssertionError("No external delivery permitted")
    import urllib.request
    import src.notifications
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(src.notifications, "send_message", forbidden)
    experiment(lab[0], lab[1], lab[4])


def test_effective_cost_can_vary_within_same_severity(lab):
    labels = labels_from_frame(lab[1], lab[4]["miss_cost_by_severity"])
    costs = {}
    for label in labels:
        if label.incident_id:
            costs.setdefault(label.severity, set()).add(label.miss_cost)
    assert any(len(v) > 1 for v in costs.values())


def test_missing_scheduled_rows_and_overlapping_partitions_rejected(lab):
    obs, labels, _, _, config = lab
    with pytest.raises(ValueError, match="каждый календарный день"):
        experiment(obs.drop(index=obs.index[250]), labels, config)
    changed = obs.copy()
    changed.loc[changed.index[-1], "split"] = "validation"
    with pytest.raises(ValueError, match="раньше TEST"):
        experiment(changed, labels, config)


def test_repeated_experiments_and_all_artifacts_are_byte_identical(lab, tmp_path):
    a = experiment(lab[0], lab[1], lab[4])
    b = experiment(lab[0], lab[1], lab[4])
    assert a == b
    x, y = tmp_path / "first", tmp_path / "second"
    save_results(x, *a)
    save_results(y, *b)
    assert len(list(x.iterdir())) == 7
    for p in x.iterdir():
        assert p.read_bytes() == (y / p.name).read_bytes(), p.name
    journal = [json.loads(line) for line in (x / "incident_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(journal) == 180
    assert all(r["truth_synthetic"] and "snapshot_at" in r["evidence"] for r in journal)


def test_cli_validation_only_and_no_send_flag(lab, tmp_path):
    source = tmp_path / "data"
    save_dataset(source)
    output = tmp_path / "out"
    args = [sys.executable, "-m", "scripts.evaluate_alerts", "--observations", str(source/"observations.csv"),
            "--labels", str(source/"labels.csv"), "--output", str(output), "--validation-only"]
    result = subprocess.run(args, capture_output=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    report = json.loads((output/"alert_report.json").read_text(encoding="utf-8"))
    assert not report["test_evaluated"] and "test_metrics" not in report
    assert "src/alerts.py" in report["reproducibility"]["code_sha256"]
    result = subprocess.run([*args, "--send"], capture_output=True)
    assert result.returncode != 0
