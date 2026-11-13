"""CLI example: carsel-train --csv /path/to/cardio_train.csv --age-unit days"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_dataset
from .store import model_path, save_artifacts, archive_run
from .training import TrainingSettings, train_carsel


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train CaRSEL on real labeled CSV data")
    parser.add_argument("--csv", required=True, help="Training CSV path")
    parser.add_argument("--age-unit", choices=["years", "days"], required=True,
                        help="Age unit in source CSV (API always uses years)")
    parser.add_argument("--target", default="cardio")
    parser.add_argument("--drop-invalid", action="store_true")
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--min-residual", type=int, default=30)
    parser.add_argument("--max-exit-error", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    df, data_summary = load_dataset(args.csv, target=args.target,
                                    age_unit=args.age_unit, drop_invalid=args.drop_invalid)
    model, report = train_carsel(df, TrainingSettings(
        target=args.target, seed=args.seed, max_layers=args.max_layers,
        min_residual=args.min_residual, max_exit_error=args.max_exit_error))
    report["ingestion"] = vars(data_summary)
    destination = args.out or model_path()
    save_artifacts(model, report, destination)
    run_id = archive_run(model, report, destination)
    print(f"Model saved at {destination}")
    print(f"Archived experiment: {run_id}")
    print(json.dumps(report["holdout_metrics"], indent=2, ensure_ascii=False))
    print("Research only; validate independently before any clinical investigation.")


if __name__ == "__main__":
    main()
