"""Generate a reproducible SYNTHETIC incident lab, not real business labels."""
import argparse
import json
from pathlib import Path

from src.synthetic_incidents import SyntheticConfig, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="SYNTHETIC Business Pulse incident lab")
    parser.add_argument("--output", default="data/synthetic_incidents")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--config", default="configs/alert_v06.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = save_dataset(Path(args.output), SyntheticConfig(seed=args.seed),
                            miss_cost_by_severity=config["miss_cost_by_severity"])
    print(manifest["source"])
    print("Созданы observations.csv, labels.csv, latent_daily.csv, manifest.json в", args.output)


if __name__ == "__main__":
    main()
