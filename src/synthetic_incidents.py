"""Explicitly synthetic interventions and availability times, never real truth.

Labels originate from simulated interventions, NOT thresholded forecast errors.
The forecasting algorithm is the median already selected in v0.5; no fitting or
selection on the new synthetic test. Partition RNGs make test changes independent
of validation generation. All trading sessions here are 08:00-22:00 UTC.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import add_features


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 20261111
    train_days: int = 240
    validation_days: int = 90
    test_days: int = 90

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in asdict(self).values()):
            raise ValueError("Параметры генератора должны быть целыми неотрицательными")
        if self.train_days < 60 or min(self.validation_days, self.test_days) < 80:
            raise ValueError("Нужно >=60 train и >=80 validation/test дней для заданных сценариев")


def generate(config: SyntheticConfig | None = None, *, miss_cost_by_severity: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    config = config or SyntheticConfig()
    costs = miss_cost_by_severity or {"mild": 15, "major": 45, "critical": 150}
    if set(costs) != {"mild", "major", "critical"} or any(
        isinstance(c, bool) or not np.isfinite(c) or c < 0 for c in costs.values()
    ):
        raise ValueError("Нужны конечные неотрицательные стоимости mild/major/critical")
    total = config.train_days + config.validation_days + config.test_days
    dates = pd.date_range("2024-01-01", periods=total, freq="D")
    values = np.empty(total)
    truth = [{"stream_id": "synthetic_shop", "date": str(d.date()), "incident_id": "",
              "severity": "", "cost_multiplier": 0., "synthetic": True} for d in dates]
    splits = []
    cursor = 0
    for part, (name, size) in enumerate((("train", config.train_days), ("validation", config.validation_days), ("test", config.test_days))):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, part]))
        end = cursor + size
        splits.extend([name] * size)
        for i in range(cursor, end):
            base = [1000, 1100, 1150, 1200, 1500, 900, 450][dates[i].day_of_week] + .8 * i
            benign = rng.uniform(.55, .8) if rng.random() < .12 else 1.
            values[i] = max(0., base * max(.3, rng.normal(1, .13)) * benign)
        start, serial = cursor + 5, 0
        while start < end - 5:
            duration = int(rng.integers(1, 4))
            severity = str(rng.choice(["mild", "major", "critical"], p=[.35, .4, .25]))
            low, high = {"mild": (.15, .28), "major": (.35, .55), "critical": (.65, .85)}[severity]
            reduction = rng.uniform(low, high)
            multiplier = float(rng.choice([.75, 1., 1.25]))
            identifier = f"SYN-{name}-{serial:03d}"
            for i in range(start, min(start + duration, end)):
                values[i] *= 1 - reduction
                truth[i].update(incident_id=identifier, severity=severity, cost_multiplier=multiplier)
            start += duration + int(rng.integers(5, 11))
            serial += 1
        cursor = end
    values = np.round(values, 2)
    daily = pd.DataFrame({"date": dates, "revenue": values})
    features = add_features(daily).set_index("date")
    origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    availability = [origin + timedelta(days=i + 1, minutes=30) for i in range(total)]
    faults = {}
    for start, size in ((0, config.train_days), (config.train_days, config.validation_days),
                        (config.train_days + config.validation_days, config.test_days)):
        for offset, kind in ((8, "incomplete"), (19, "stale"), (28, "missing_actual"),
                             (59, "early_forecast"), (70, "future_source_time")):
            if offset < size:
                faults[start + offset] = kind
                if kind in {"incomplete", "stale", "missing_actual"}:
                    availability[start + offset] += timedelta(hours=3)
    observations = []
    for i in range(28, total):
        day = origin + timedelta(days=i)
        forecast_at = day + timedelta(hours=1)
        if faults.get(i) == "early_forecast":
            forecast_at = day  # previous day data only ready at 00:30
        history_ready_at = max(availability[i-28:i])
        history_ready = history_ready_at <= forecast_at
        # Never expose a forecast computed from unavailable historical actuals.
        forecast = float(features.loc[dates[i], "baseline"]) if history_ready else None
        snapshot = day + timedelta(days=1, hours=1, minutes=30)
        actual_ready = availability[i] <= snapshot
        actual = float(values[i])
        data_through = str(dates[i].date())
        updated = availability[i] + timedelta(minutes=15) if actual_ready else snapshot - timedelta(minutes=30)
        fault = faults.get(i, "none")
        if fault == "incomplete":
            actual *= .5
        elif fault == "missing_actual":
            actual = None
        elif fault == "stale":
            actual = float(values[i-1])
            data_through = str(dates[i-1].date())
            updated = day - timedelta(hours=12)
        elif fault == "future_source_time":
            updated = snapshot + timedelta(hours=1)
        observations.append({
            "split": splits[i], "stream_id": "synthetic_shop", "date": str(dates[i].date()),
            "snapshot_id": f"SYN-snapshot-{i:04d}", "forecast": forecast, "actual": actual,
            "forecast_at": forecast_at.isoformat(),
            "history_closed_at": (day - timedelta(hours=2)).isoformat(),
            "history_complete_at": history_ready_at.isoformat() if history_ready else None,
            "history_through": str((dates[i] - pd.Timedelta(days=1 if history_ready else 2)).date()),
            "history_complete": history_ready,
            "day_open_at": (day + timedelta(hours=8)).isoformat(),
            "day_closed_at": (day + timedelta(hours=22)).isoformat(),
            "actual_complete": actual_ready, "actual_complete_at": availability[i].isoformat() if actual_ready else None,
            "data_through": data_through, "source_updated_at": updated.isoformat(),
            "snapshot_at": snapshot.isoformat(), "forecast_method": "v05_fixed_four_week_median",
            "synthetic_quality_scenario": fault,
        })
    label_frame = pd.DataFrame(truth).iloc[28:].copy()
    label_frame.insert(0, "split", splits[28:])
    daily["available_at"] = [v.isoformat() for v in availability]
    daily["split"] = splits
    manifest = {
        "source": "SYNTHETIC educational simulation; not real customer incidents",
        "config": asdict(config), "miss_cost_by_severity": costs,
        "forecast_method": "v0.5 selected four-week median, frozen before this experiment",
        "label_mechanism": "exogenous simulated service interventions, independent of residual thresholds",
        "trading_calendar": "08:00-22:00 UTC every day; late actuals complete after forecast cutoff",
        "split_dates": {name: [str(daily.loc[daily.split.eq(name), 'date'].min().date()),
                               str(daily.loc[daily.split.eq(name), 'date'].max().date())]
                        for name in ("train", "validation", "test")},
        "limitations": "Invented incident mix, times and costs; same-generator test is not commercial validation.",
    }
    return pd.DataFrame(observations), label_frame, daily, manifest


def save_dataset(output: Path, config: SyntheticConfig | None = None, *, miss_cost_by_severity: dict | None = None) -> dict:
    observations, labels, daily, manifest = generate(config, miss_cost_by_severity=miss_cost_by_severity)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("observations.csv", observations), ("labels.csv", labels), ("latent_daily.csv", daily)):
        frame.to_csv(output / name, index=False, lineterminator="\n")
    manifest["sha256"] = {name: sha256((output/name).read_bytes()).hexdigest()
                          for name in ("observations.csv", "labels.csv", "latent_daily.csv")}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
