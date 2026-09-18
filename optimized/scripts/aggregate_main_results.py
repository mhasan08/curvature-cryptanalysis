"""Aggregate and validate the six shared-stencil trained-model runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = {
    "direction_recovery": ("dir_matched_mean", 1.0),
    "hit_0_90": ("dir_frac_gt_090", 1.0),
    "victim_accuracy_pct": ("victim_acc", 100.0),
    "surrogate_accuracy_pct": ("surrogate_acc", 100.0),
    "agreement_pct": ("agreement", 100.0),
    "kl_divergence": ("kl", 1.0),
    "centered_logit_mse": ("centered_logit_mse", 1.0),
    "ffn_cka": ("ffn_cka", 1.0),
    "matched_feature_correlation": ("feature_corr_matched_mean", 1.0),
}

EXPECTED = {
    "T": 16,
    "probe_locations": 1,
    "projections_per_probe": 16,
    "hessian_mode": "finite_diff",
    "fd_eps": 0.01,
    "dict_init": "random",
    "dict_restarts": 3,
    "dict_steps": 3000,
    "seed": 0,
    "structural_query_count": 8193,
    "completion_query_count": 12000,
    "total_query_count": 20193,
}


def load_runs(root: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {"gelu": [], "silu": []}
    paths = sorted(root.glob("*/result.json"))
    if len(paths) != 6:
        raise ValueError(f"Expected six result records under {root}, found {len(paths)}")

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        activation = record.get("activation")
        if activation not in grouped:
            raise ValueError(f"Unexpected activation in {path}: {activation!r}")
        for key, expected in EXPECTED.items():
            actual = record.get(key)
            if actual != expected:
                raise ValueError(
                    f"Configuration mismatch in {path}: {key}={actual!r}, "
                    f"expected {expected!r}"
                )
        grouped[activation].append(record)

    for activation, records in grouped.items():
        if len(records) != 3:
            raise ValueError(f"Expected three {activation} records, found {len(records)}")
    return grouped


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def aggregate(activation: str, records: list[dict]) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "activation": activation,
        "num_victim_seeds": len(records),
    }
    for label, (key, scale) in METRICS.items():
        values = [float(record[key]) * scale for record in records]
        mean, std = mean_std(values)
        row[f"{label}_mean"] = mean
        row[f"{label}_std"] = std

    drops = [
        (float(record["victim_acc"]) - float(record["surrogate_acc"])) * 100.0
        for record in records
    ]
    drop_mean, drop_std = mean_std(drops)
    row["accuracy_drop_pp_mean"] = drop_mean
    row["accuracy_drop_pp_std"] = drop_std
    row["structural_queries"] = EXPECTED["structural_query_count"]
    row["completion_queries"] = EXPECTED["completion_query_count"]
    row["total_queries"] = EXPECTED["total_query_count"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/shared_runs"),
        help="Directory containing one subdirectory per run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/main_results.csv"),
        help="Destination CSV path",
    )
    args = parser.parse_args()

    grouped = load_runs(args.input_root)
    rows = [aggregate(activation, grouped[activation]) for activation in ("gelu", "silu")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['activation']}: "
            f"DirRec={row['direction_recovery_mean']:.4f} +/- "
            f"{row['direction_recovery_std']:.4f}, "
            f"agreement={row['agreement_pct_mean']:.2f} +/- "
            f"{row['agreement_pct_std']:.2f}%, "
            f"queries={row['total_queries']}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
