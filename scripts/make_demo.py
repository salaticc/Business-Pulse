"""Deterministic SYNTHETIC demo. Not real sales and not evidence of business demand."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
dates = pd.date_range("2025-01-01", periods=280, freq="D")
rows = []
for i, d in enumerate(dates):
    weekday_factor = [1.00, 1.05, 1.15, 1.08, 1.35, 0.84, 0.25][d.dayofweek]
    orders = max(0, int(rng.normal(34 * weekday_factor + 0.025 * i, 6)))
    if i in {165, 231, 255}:
        orders = int(orders * 0.16)  # injected artificial incidents
    aov = max(10.0, rng.normal(36.0, 2.0))
    revenue = round(orders * aov, 2)
    rows.append({"date": str(d.date()), "revenue": revenue, "orders": orders})

out = Path("data/demo/daily_sales.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Создано {out} — только искусственные данные; инциденты искусственные")
