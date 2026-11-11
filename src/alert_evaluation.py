"""Synthetic alert evaluation. Truth is joined only AFTER causal decisions.

Daily confusion and incident-level loss have deliberately different units:
one missed incident incurs one label-defined cost; every false emitted
business warning incurs a review cost. Data-quality warnings are separate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
import math

from src.alerts import AlertMonitor, AlertPolicy, Observation, _amount


@dataclass(frozen=True)
class IncidentLabel:
    stream_id: str
    date: str
    incident_id: str | None = None
    severity: str | None = None
    miss_cost: float = 0
    synthetic: bool = True

    def __post_init__(self) -> None:
        date.fromisoformat(self.date)
        if not self.stream_id or not _amount(self.miss_cost):
            raise ValueError("Нужны stream_id и конечная неотрицательная miss_cost")
        if self.synthetic is not True:
            raise ValueError("Этот учебный evaluator принимает только явно синтетическую разметку")
        if self.incident_id:
            if not self.severity:
                raise ValueError("У инцидента должна быть задана тяжесть")
        elif self.severity is not None or self.miss_cost != 0:
            raise ValueError("У отрицательного примера нет тяжести или стоимости пропуска")


def confusion(truth: list[bool], predicted: list[bool]) -> dict:
    if not truth or len(truth) != len(predicted):
        raise ValueError("Нужны непустые массивы одинаковой длины")
    tp = sum(t and p for t, p in zip(truth, predicted))
    fp = sum(not t and p for t, p in zip(truth, predicted))
    fn = sum(t and not p for t, p in zip(truth, predicted))
    tn = sum(not t and not p for t, p in zip(truth, predicted))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def score_alerts(decisions: list[dict], labels: list[IncidentLabel], false_alert_cost: float) -> tuple[dict, list[dict], list[dict]]:
    if not _amount(false_alert_cost):
        raise ValueError("false_alert_cost должен быть конечным неотрицательным числом")
    label_map = {(x.stream_id, x.date): x for x in labels}
    keys = [(r["stream_id"], r["date"]) for r in decisions]
    if not keys or len(label_map) != len(labels) or len(keys) != len(set(keys)) or set(keys) != set(label_map):
        raise ValueError("Журнал и разметка должны содержать ровно одну строку на одну и ту же stream/date")
    truth, candidate, emitted = [], [], []
    journal, incidents = [], {}
    for row in decisions:
        label = label_map[(row["stream_id"], row["date"])]
        positive = bool(label.incident_id)
        warning = row["event_type"] == "revenue_drop" and row["emitted"]
        truth.append(positive)
        candidate.append(row["business_candidate"])
        emitted.append(warning)
        journal.append({**deepcopy(row), "truth_synthetic": True,
                        "confirmed_in_synthetic_truth": positive,
                        "truth_incident_id": label.incident_id, "truth_severity": label.severity,
                        "truth_miss_cost": label.miss_cost})
        if positive:
            incidents.setdefault((label.stream_id, label.incident_id), []).append((label, warning))
    ledger = []
    for (stream, identifier), members in sorted(incidents.items()):
        members.sort(key=lambda pair: pair[0].date)
        first = members[0][0]
        if any(m.severity != first.severity or m.miss_cost != first.miss_cost for m, _ in members):
            raise ValueError("Тяжесть и стоимость одного инцидента должны совпадать на всех его днях")
        dates = [date.fromisoformat(m.date) for m, _ in members]
        if any(b != a + timedelta(days=1) for a, b in zip(dates, dates[1:])):
            raise ValueError("Один incident_id должен описывать непрерывный эпизод")
        warnings = [m.date for m, warned in members if warned]
        ledger.append({
            "stream_id": stream, "incident_id": identifier,
            "start_date": first.date, "end_date": members[-1][0].date,
            "severity": first.severity, "miss_cost": first.miss_cost,
            "detected": bool(warnings), "business_warnings": len(warnings),
            "first_warning_date": warnings[0] if warnings else None,
            "detection_delay_days": (date.fromisoformat(warnings[0]) - dates[0]).days if warnings else None,
            "incurred_miss_cost": 0. if warnings else float(first.miss_cost),
        })
    daily = confusion(truth, emitted)
    detected = sum(r["detected"] for r in ledger)
    missed = len(ledger) - detected
    false_cost = float(daily["fp"] * false_alert_cost)
    miss_cost = float(math.fsum(r["incurred_miss_cost"] for r in ledger))
    metrics = {
        "daily_emitted": daily,
        "daily_candidates": confusion(truth, candidate),
        "incidents": {"tp": detected, "fn": missed, "total": len(ledger),
                      "recall": detected / len(ledger) if ledger else None},
        "false_business_warnings": daily["fp"],
        "false_alert_cost": false_cost, "missed_incident_cost": miss_cost,
        "total_error_cost": false_cost + miss_cost,
        "cost_unit": "synthetic decision units; not measured financial loss",
        "scheduled_days": len(decisions),
        "eligible_days": sum(r["eligible"] for r in decisions),
        "blocked_days": sum(not r["eligible"] for r in decisions),
        "business_warnings": sum(emitted),
        "suppressed_business_candidates": sum(r["business_candidate"] and r["suppressed"] for r in decisions),
        "data_quality_warnings": sum(r["event_type"] == "data_quality" and r["emitted"] for r in decisions),
    }
    return metrics, ledger, journal


@dataclass
class PolicySelection:
    policy: AlertPolicy
    state: dict
    metrics: dict
    comparison: list[dict]
    journal: list[dict]
    incidents: list[dict]


def select_policy(
    validation_observations: list[Observation], validation_labels: list[IncidentLabel],
    policies: list[AlertPolicy], false_alert_cost: float,
) -> PolicySelection:
    """The function cannot access test observations, labels, severity or costs."""
    if not policies or len({p.identity for p in policies}) != len(policies):
        raise ValueError("Нужен непустой список разных политик")
    rows, best, best_key = [], None, None
    for index, policy in enumerate(policies):
        monitor = AlertMonitor(policy)
        metrics, ledger, journal = score_alerts(monitor.run(validation_observations), validation_labels, false_alert_cost)
        if metrics["eligible_days"] == 0:
            raise ValueError("Нет пригодных validation-наблюдений; выбирать порог нельзя")
        row = {
            "policy_index": index, "relative_threshold_pct": policy.relative_drop_pct,
            "absolute_threshold": policy.absolute_shortfall,
            **metrics["daily_emitted"],
            "candidate_recall": metrics["daily_candidates"]["recall"],
            "incident_tp": metrics["incidents"]["tp"], "incident_fn": metrics["incidents"]["fn"],
            "incident_recall": metrics["incidents"]["recall"],
            "false_business_warnings": metrics["false_business_warnings"],
            "false_alert_cost": metrics["false_alert_cost"],
            "missed_incident_cost": metrics["missed_incident_cost"],
            "total_error_cost": metrics["total_error_cost"],
        }
        rows.append(row)
        key = (metrics["total_error_cost"], metrics["false_business_warnings"], index)
        if best_key is None or key < best_key:
            best_key = key
            best = PolicySelection(policy, deepcopy(monitor.state), metrics, rows, journal, ledger)
    best.comparison = rows
    return best
