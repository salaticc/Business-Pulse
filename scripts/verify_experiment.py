"""Regression on a published synthetic fixture, not independent validation."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify numerical synthetic alert regression")
    parser.add_argument("--report", required=True)
    parser.add_argument("--reference", default="reports/v0.6.0/alert_report.json")
    args = parser.parse_args()
    actual = json.loads(Path(args.report).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    for field in ("config", "selected_policy", "validation_dates", "test_dates", "threshold_comparison",
                  "validation_metrics", "test_metrics"):
        if actual.get(field) != reference.get(field):
            raise SystemExit(f"Numerical regression differs in {field}; inspect without tuning on TEST.")
    print("Synthetic numerical regression matches v0.6.0; not new independent validation.")


if __name__ == "__main__":
    main()
