"""Local, stateful alert decisions. No labels, costs, transport or network access.

An observation is a completed-day snapshot with explicit availability evidence.
Availability metadata must come from the source; it cannot be inferred from sales.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path


def timestamp(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("Время должно быть строкой ISO с часовым поясом или None")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Все временные отметки должны содержать часовой пояс")
    return result.astimezone(timezone.utc)


def _amount(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


@dataclass(frozen=True)
class AlertPolicy:
    relative_drop_pct: float = 25
    absolute_shortfall: float = 250
    cooldown_days: int = 7
    max_data_age_hours: float = 24
    high_drop_pct: float = 40
    critical_drop_pct: float = 65

    def __post_init__(self) -> None:
        for name in ("relative_drop_pct", "absolute_shortfall", "max_data_age_hours", "high_drop_pct", "critical_drop_pct"):
            if not _amount(getattr(self, name)):
                raise ValueError(f"{name}: нужно конечное неотрицательное число")
        if not 0 <= self.relative_drop_pct <= 100:
            raise ValueError("relative_drop_pct должен быть от 0 до 100")
        if not 0 <= self.high_drop_pct <= self.critical_drop_pct <= 100:
            raise ValueError("Уровни серьёзности должны возрастать в пределах 0..100")
        if type(self.cooldown_days) is not int or self.cooldown_days < 0:
            raise ValueError("cooldown_days должен быть целым числом >=0")
        if self.max_data_age_hours == 0:
            raise ValueError("max_data_age_hours должен быть >0")

    @property
    def identity(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Observation:
    stream_id: str
    date: str
    snapshot_id: str
    forecast: float | None
    actual: float | None
    forecast_at: str | None
    history_closed_at: str | None
    history_complete_at: str | None
    history_through: str | None
    history_complete: bool
    day_open_at: str
    day_closed_at: str
    actual_complete: bool
    actual_complete_at: str | None
    data_through: str | None
    source_updated_at: str | None
    snapshot_at: str
    forecast_method: str = "external"

    def __post_init__(self) -> None:
        if not self.stream_id or not self.snapshot_id:
            raise ValueError("Нужны stream_id и snapshot_id")
        date.fromisoformat(self.date)
        for name in ("history_through", "data_through"):
            if getattr(self, name):
                date.fromisoformat(getattr(self, name))
        for name in ("forecast_at", "history_closed_at", "history_complete_at", "day_open_at", "day_closed_at",
                     "actual_complete_at", "source_updated_at", "snapshot_at"):
            parsed = timestamp(getattr(self, name))
            if name in {"day_open_at", "day_closed_at", "snapshot_at"} and parsed is None:
                raise ValueError(f"Не задано {name}")
        for name in ("history_complete", "actual_complete"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} должен быть bool, неизвестная полнота не является True")


def quality_reasons(obs: Observation, policy: AlertPolicy) -> list[str]:
    """Fail closed for business decisions; identify the unavailable evidence."""
    reasons = []
    t = {name: timestamp(getattr(obs, name)) for name in (
        "forecast_at", "history_closed_at", "history_complete_at", "day_open_at", "day_closed_at",
        "actual_complete_at", "source_updated_at", "snapshot_at",
    )}
    previous = str(date.fromisoformat(obs.date) - timedelta(days=1))
    if datetime.fromisoformat(obs.day_open_at).date().isoformat() != obs.date or datetime.fromisoformat(obs.day_closed_at).date().isoformat() != obs.date:
        reasons.append("calendar_session_date_mismatch")
    if obs.history_closed_at and datetime.fromisoformat(obs.history_closed_at).date().isoformat() != previous:
        reasons.append("history_closure_date_mismatch")
    if not obs.history_complete or obs.history_through != previous:
        reasons.append("forecast_history_incomplete")
    if not all(t[name] is not None for name in ("forecast_at", "history_closed_at", "history_complete_at")):
        reasons.append("forecast_availability_unknown")
    elif not t["history_closed_at"] <= t["history_complete_at"] <= t["forecast_at"] < t["day_open_at"]:
        reasons.append("forecast_before_history_ready_or_after_open")
    if not t["day_open_at"] < t["day_closed_at"] <= t["snapshot_at"]:
        reasons.append("target_day_not_closed_or_invalid_calendar")
    if not obs.actual_complete:
        reasons.append("actual_incomplete")
    if obs.data_through != obs.date:
        reasons.append("wrong_data_date")
    if t["actual_complete_at"] is None or not t["day_closed_at"] <= t["actual_complete_at"] <= t["snapshot_at"]:
        reasons.append("actual_completion_not_available_as_of_snapshot")
    updated = t["source_updated_at"]
    if updated is None:
        reasons.append("source_update_unknown")
    elif updated > t["snapshot_at"]:
        reasons.append("source_timestamp_in_future")
    else:
        if (t["snapshot_at"] - updated).total_seconds() > policy.max_data_age_hours * 3600:
            reasons.append("source_stale")
        if t["actual_complete_at"] is not None and updated < t["actual_complete_at"]:
            reasons.append("source_update_before_completion")
    if not _amount(obs.forecast):
        reasons.append("invalid_or_missing_forecast")
    if not _amount(obs.actual):
        reasons.append("invalid_or_missing_actual")
    return reasons


class AlertMonitor:
    """Sequential local decisions, keyed by stream and calendar date.

    Exact replay is suppressed. A revised date requires explicit replay from
    an earlier clean state, avoiding silently inconsistent episode histories.
    State belongs to one policy; label or cost data never enter this class.
    """

    def __init__(self, policy: AlertPolicy, state: dict | None = None):
        self.policy = policy
        self.state = deepcopy(state) if state is not None else {
            "version": 1, "policy_sha256": policy.identity,
            "last_dates": {}, "last_snapshots": {}, "active": {}, "processed": {},
        }
        if self.state.get("version") != 1 or self.state.get("policy_sha256") != policy.identity:
            raise ValueError("Состояние относится к другой версии или политике")

    def process(self, obs: Observation) -> dict:
        evidence = asdict(obs)
        for name in ("forecast", "actual"):
            if isinstance(evidence[name], Real) and not math.isfinite(evidence[name]):
                evidence[name] = str(evidence[name])
        fingerprint = sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()
        key = json.dumps([obs.stream_id, obs.date], ensure_ascii=False)
        if key in self.state["processed"]:
            old = self.state["processed"][key]
            if old["fingerprint"] != fingerprint:
                raise ValueError("Изменён уже обработанный снимок; требуется явный replay с чистым состоянием")
            return {**deepcopy(old["decision"]), "emitted": False, "suppressed": old["decision"]["event_type"] != "none",
                    "suppression_reason": "already_processed"}
        previous = self.state["last_dates"].get(obs.stream_id)
        if previous is not None and obs.date <= previous:
            raise ValueError("Наблюдения одного потока должны поступать по возрастанию дат")
        last_snapshot = self.state["last_snapshots"].get(obs.stream_id)
        if last_snapshot is not None and timestamp(obs.snapshot_at) < timestamp(last_snapshot):
            raise ValueError("Время снимка не должно двигаться назад")
        reasons = quality_reasons(obs, self.policy)
        eligible = not reasons
        shortfall = max(float(obs.forecast) - float(obs.actual), 0.) if eligible else None
        relative = 100 * shortfall / obs.forecast if eligible and obs.forecast > 0 else (0. if eligible else None)
        candidate = bool(eligible and shortfall > 0 and shortfall >= self.policy.absolute_shortfall
                         and relative >= self.policy.relative_drop_pct)
        kind, severity = "none", "none"
        if reasons:
            kind, severity = "data_quality", "high"
        elif candidate:
            kind = "revenue_drop"
            severity = "critical" if relative >= self.policy.critical_drop_pct else (
                "high" if relative >= self.policy.high_drop_pct else "warning")
            reasons = ["absolute_shortfall_at_or_above_threshold", "relative_drop_at_or_above_threshold"]
        decision = {
            "stream_id": obs.stream_id, "date": obs.date, "snapshot_id": obs.snapshot_id,
            "snapshot_at": obs.snapshot_at, "event_type": kind, "severity": severity,
            "eligible": eligible, "business_candidate": candidate,
            "forecast": evidence["forecast"], "actual": evidence["actual"],
            "absolute_shortfall": shortfall, "relative_drop_pct": relative,
            "absolute_threshold": self.policy.absolute_shortfall,
            "relative_threshold_pct": self.policy.relative_drop_pct,
            "reasons": reasons, "evidence": evidence,
            "episode_id": None, "emitted": False, "suppressed": False, "suppression_reason": "",
        }
        active = self.state["active"].setdefault(obs.stream_id, {})
        for other in list(active):
            if other != kind:
                del active[other]
        if kind != "none":
            current = date.fromisoformat(obs.date)
            episode = active.get(kind)
            signature = reasons if kind == "data_quality" else []
            contiguous = episode is not None and current == date.fromisoformat(episode["last_date"]) + timedelta(days=1)
            if not contiguous or episode["signature"] != signature:
                episode = {"start_date": obs.date, "last_emit": None, "highest_severity": 0, "signature": signature}
            rank = {"warning": 1, "high": 2, "critical": 3}[severity]
            elapsed = (current - date.fromisoformat(episode["last_emit"])).days if episode["last_emit"] else None
            emit = elapsed is None or elapsed >= self.policy.cooldown_days or rank > episode["highest_severity"]
            decision.update({
                "episode_id": json.dumps([obs.stream_id, kind, episode["start_date"]], ensure_ascii=False),
                "emitted": emit, "suppressed": not emit,
                "suppression_reason": "" if emit else "ongoing_episode_within_cooldown",
            })
            if emit:
                episode["last_emit"] = obs.date
            episode["highest_severity"] = max(rank, episode["highest_severity"])
            episode["last_date"] = obs.date
            active[kind] = episode
        self.state["last_dates"][obs.stream_id] = obs.date
        self.state["last_snapshots"][obs.stream_id] = obs.snapshot_at
        self.state["processed"][key] = {"fingerprint": fingerprint, "decision": deepcopy(decision)}
        return decision

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load_state(cls, path: str | Path, policy: AlertPolicy) -> AlertMonitor:
        return cls(policy, json.loads(Path(path).read_text(encoding="utf-8")))

    def run(self, observations: list[Observation]) -> list[dict]:
        keys = [(o.stream_id, o.date) for o in observations]
        if len(keys) != len(set(keys)):
            raise ValueError("Одна строка на stream_id/date; повторы передавайте отдельным replay")
        return [self.process(o) for o in sorted(observations, key=lambda o: (o.date, o.stream_id))]
