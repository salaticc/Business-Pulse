from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json

import numpy as np
import pytest

from src.alerts import AlertMonitor, AlertPolicy, Observation
from src.alert_evaluation import IncidentLabel, confusion, score_alerts, select_policy


def observation(day=0, actual=600., forecast=1000., **changes):
    d = datetime(2025, 1, 2, tzinfo=timezone.utc) + timedelta(days=day)
    row = Observation(
        stream_id="shop", date=str(d.date()), snapshot_id=f"snapshot-{day}", forecast=forecast, actual=actual,
        forecast_at=(d+timedelta(hours=1)).isoformat(),
        history_closed_at=(d-timedelta(hours=2)).isoformat(),
        history_complete_at=(d+timedelta(minutes=30)).isoformat(),
        history_through=str((d-timedelta(days=1)).date()), history_complete=True,
        day_open_at=(d+timedelta(hours=8)).isoformat(), day_closed_at=(d+timedelta(hours=22)).isoformat(),
        actual_complete=True, actual_complete_at=(d+timedelta(days=1, minutes=30)).isoformat(),
        data_through=str(d.date()), source_updated_at=(d+timedelta(days=1, hours=1)).isoformat(),
        snapshot_at=(d+timedelta(days=1, hours=1, minutes=30)).isoformat(),
    )
    return replace(row, **changes)


def label(obs, identifier=None, cost=0, severity=None):
    return IncidentLabel(obs.stream_id, obs.date, identifier, severity, cost)


def test_thresholds_are_joint_and_inclusive_and_not_clipped_to_one():
    policy = AlertPolicy(25, .1)
    row = AlertMonitor(policy).process(observation(actual=.3, forecast=.4))
    assert row["relative_drop_pct"] == pytest.approx(25.)
    assert row["emitted"]  # actual shortfall .10000000000000003 >= .1
    assert not AlertMonitor(AlertPolicy(25, 400.01)).process(observation())["emitted"]
    assert not AlertMonitor(AlertPolicy(40.01, 400)).process(observation())["emitted"]
    assert AlertMonitor(AlertPolicy(40, 400)).process(observation())["emitted"]
    assert not AlertMonitor(AlertPolicy(0, 0)).process(observation(actual=1000))["emitted"]


@pytest.mark.parametrize("forecast,actual", [(0, 0), (0, 10), (1000, 1200)])
def test_zero_forecast_and_positive_deviations_are_not_revenue_drops(forecast, actual):
    row = AlertMonitor(AlertPolicy(0, 0)).process(observation(forecast=forecast, actual=actual))
    assert row["eligible"] and row["relative_drop_pct"] == 0 and not row["emitted"]


@pytest.mark.parametrize("changes,reason", [
    ({"history_complete": False}, "forecast_history_incomplete"),
    ({"history_through": "2025-01-02"}, "forecast_history_incomplete"),
    ({"forecast_at": "2025-01-01T18:00:00+00:00"}, "forecast_before_history_ready_or_after_open"),
    ({"forecast_at": "2025-01-02T09:00:00+00:00"}, "forecast_before_history_ready_or_after_open"),
    ({"history_complete_at": None}, "forecast_availability_unknown"),
    ({"actual_complete": False}, "actual_incomplete"),
    ({"actual_complete_at": "2025-01-03T04:00:00+00:00"}, "actual_completion_not_available_as_of_snapshot"),
    ({"data_through": "2025-01-01"}, "wrong_data_date"),
    ({"source_updated_at": "2025-01-01T00:00:00+00:00"}, "source_stale"),
    ({"source_updated_at": "2025-01-03T03:00:00+00:00"}, "source_timestamp_in_future"),
    ({"source_updated_at": None}, "source_update_unknown"),
    ({"actual": None}, "invalid_or_missing_actual"),
    ({"actual": np.nan}, "invalid_or_missing_actual"),
    ({"forecast": np.inf}, "invalid_or_missing_forecast"),
    ({"actual": -1}, "invalid_or_missing_actual"),
    ({"snapshot_at": "2025-01-02T18:00:00+00:00"}, "target_day_not_closed_or_invalid_calendar"),
    ({"day_open_at": "2025-01-01T08:00:00+00:00"}, "calendar_session_date_mismatch"),
])
def test_quality_gate_blocks_business_classification(changes, reason):
    row = AlertMonitor(AlertPolicy()).process(observation(**changes))
    assert row["event_type"] == "data_quality" and row["emitted"]
    assert not row["eligible"] and not row["business_candidate"]
    assert row["absolute_shortfall"] is None and reason in row["reasons"]
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("changes", [{"snapshot_at": "2025-01-03T01:30:00"}, {"history_complete": "False"}])
def test_ambiguous_metadata_rejected(changes):
    with pytest.raises(ValueError):
        observation(**changes)


def test_episode_dedup_escalation_reminder_and_resolution():
    monitor = AlertMonitor(AlertPolicy(20, 100, cooldown_days=7))
    rows = monitor.run([observation(i, actual=700 if i < 2 else 300) for i in range(10)])
    assert [i for i, r in enumerate(rows) if r["emitted"]] == [0, 2, 9]
    assert rows[0]["severity"] == "warning" and rows[2]["severity"] == "critical"
    assert len({r["episode_id"] for r in rows}) == 1
    assert monitor.process(observation(10, actual=1000))["event_type"] == "none"
    fresh = monitor.process(observation(11, actual=700))
    assert fresh["emitted"] and fresh["episode_id"] != rows[0]["episode_id"]


def test_streams_quality_and_business_states_are_separate():
    monitor = AlertMonitor(AlertPolicy())
    a = monitor.process(observation())
    b = monitor.process(observation(stream_id="other"))
    assert a["emitted"] and b["emitted"] and a["episode_id"] != b["episode_id"]
    q = monitor.process(observation(1, actual_complete=False))
    assert q["event_type"] == "data_quality" and q["emitted"]
    again = monitor.process(observation(2))
    assert again["emitted"] and again["episode_id"] != a["episode_id"]


def test_state_survives_restart_exact_replay_is_suppressed_and_revisions_rejected(tmp_path):
    policy = AlertPolicy()
    first = observation()
    monitor = AlertMonitor(policy)
    monitor.process(first)
    path = tmp_path / "state.json"
    monitor.save_state(path)
    resumed = AlertMonitor.load_state(path, policy)
    replay = resumed.process(first)
    assert not replay["emitted"] and replay["suppression_reason"] == "already_processed"
    assert not resumed.process(observation(1))["emitted"]
    with pytest.raises(ValueError, match="Изменён"):
        resumed.process(replace(first, actual=10))
    with pytest.raises(ValueError, match="политике"):
        AlertMonitor.load_state(path, AlertPolicy(30))
    with pytest.raises(ValueError, match="возрастанию"):
        resumed.process(observation(-1))


def test_future_changes_do_not_change_previous_decisions():
    a = [observation(i) for i in range(10)]
    b = a[:5] + [observation(i, actual=0) for i in range(5, 10)]
    assert AlertMonitor(AlertPolicy()).run(a)[:5] == AlertMonitor(AlertPolicy()).run(b)[:5]


def test_known_confusion_and_costs_with_different_incident_units():
    obs = [observation(0), observation(1, actual=1000), observation(2), observation(3, actual=1000)]
    labels = [label(obs[0], "A", 20, "mild"), label(obs[1], "B", 150, "critical"), label(obs[2]), label(obs[3])]
    metrics, ledger, journal = score_alerts(AlertMonitor(AlertPolicy()).run(obs), labels, 3)
    assert metrics["daily_emitted"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": .5, "recall": .5, "f1": .5}
    assert metrics["false_alert_cost"] == 3
    assert metrics["missed_incident_cost"] == 150 and metrics["total_error_cost"] == 153
    assert metrics["incidents"] == {"tp": 1, "fn": 1, "total": 2, "recall": .5}
    assert journal[0]["confirmed_in_synthetic_truth"] and not journal[2]["confirmed_in_synthetic_truth"]


def test_suppression_changes_day_recall_but_not_episode_recall_and_cost():
    obs = [observation(i) for i in range(3)]
    labels = [label(o, "A", 45, "major") for o in obs]
    metrics, ledger, _ = score_alerts(AlertMonitor(AlertPolicy()).run(obs), labels, 3)
    assert metrics["daily_emitted"]["fn"] == 2
    assert metrics["daily_candidates"]["recall"] == 1
    assert metrics["incidents"]["recall"] == 1 and metrics["total_error_cost"] == 0
    assert ledger[0]["business_warnings"] == 1


def test_blocked_incident_still_counts_as_miss_cost():
    obs = [observation(i, actual_complete=False) for i in range(2)]
    metrics, ledger, _ = score_alerts(AlertMonitor(AlertPolicy()).run(obs), [label(o, "A", 45, "major") for o in obs], 3)
    assert metrics["blocked_days"] == 2 and metrics["daily_emitted"]["fn"] == 2
    assert metrics["missed_incident_cost"] == 45 and metrics["incidents"]["fn"] == 1
    assert metrics["false_alert_cost"] == 0  # quality warnings are a separate channel


def test_early_false_signal_can_suppress_later_incident_without_oracle_reset():
    obs = [observation(0, actual=700), observation(1, actual=700)]
    labels = [label(obs[0]), label(obs[1], "new-incident", 150, "critical")]
    decisions = AlertMonitor(AlertPolicy(20, 100)).run(obs)
    metrics, _, _ = score_alerts(decisions, labels, 3)
    assert decisions[1]["business_candidate"] and not decisions[1]["emitted"]
    assert metrics["false_alert_cost"] == 3 and metrics["missed_incident_cost"] == 150
    assert metrics["incidents"]["fn"] == 1  # no reset using hidden true incident IDs


def test_selection_minimizes_cost_not_maximum_recall():
    obs = [observation(0, actual=700), observation(1, actual=1000), observation(2, actual=700)]
    labels = [label(obs[0], "A", 1, "mild"), label(obs[1]), label(obs[2])]
    cheap_misses = select_policy(obs, labels, [AlertPolicy(20, 100), AlertPolicy(50, 100)], 3)
    assert cheap_misses.policy.relative_drop_pct == 50
    expensive_misses = select_policy(obs, [replace(labels[0], miss_cost=100), *labels[1:]], [AlertPolicy(20, 100), AlertPolicy(50, 100)], 3)
    assert expensive_misses.policy.relative_drop_pct == 20


@pytest.mark.parametrize("kwargs", [{"relative_drop_pct": 101}, {"absolute_shortfall": -1},
                                  {"cooldown_days": True}, {"max_data_age_hours": 0},
                                  {"critical_drop_pct": 20}, {"absolute_shortfall": float('nan')}])
def test_invalid_policies_rejected(kwargs):
    with pytest.raises(ValueError):
        AlertPolicy(**kwargs)


def test_undefined_metrics_are_null_not_perfect_scores():
    assert confusion([False], [False])["precision"] is None
    assert confusion([False], [False])["recall"] is None
    assert confusion([True], [False])["f1"] == 0
    with pytest.raises(ValueError):
        confusion([], [])


def test_bad_truth_and_missing_denominator_rows_rejected():
    obs = [observation()]
    decisions = AlertMonitor(AlertPolicy()).run(obs)
    with pytest.raises(ValueError):
        score_alerts(decisions, [], 3)
    with pytest.raises(ValueError):
        IncidentLabel("shop", obs[0].date, "A", "major", -1)
    with pytest.raises(ValueError):
        IncidentLabel("shop", obs[0].date, synthetic=False)
    with pytest.raises(ValueError):
        AlertMonitor(AlertPolicy()).run([obs[0], obs[0]])
