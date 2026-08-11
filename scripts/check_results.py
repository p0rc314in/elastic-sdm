#!/usr/bin/env python3
"""Extract fresh balanced-write measurements and enforce reproduction bounds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RECALL_ARMS = (
    "native_sdm_rw16_price000",
    "native_sdm_rw16_price0012",
    "native_sdm_rw16_price0024",
    "native_sdm_rw16_price0040",
    "native_sdm_rw16_price012",
    "native_sdm_rw16_price0240",
    "native_sdm_rw16_price0400",
)
LANGUAGE_ARMS = (
    "native_sdm_rw16_price000",
    "native_sdm_rw16_price0024",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing reproduction result: {path}")
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def require_close(label: str, observed: float, expected: float, tolerance: float) -> None:
    if not math.isfinite(observed) or abs(observed - expected) > tolerance:
        raise SystemExit(
            f"FAIL {label}: observed {observed:.8g}, expected {expected:.8g} "
            f"within ±{tolerance:.8g}"
        )


def require_between(label: str, observed: float, lower: float, upper: float) -> None:
    if not math.isfinite(observed) or not lower <= observed <= upper:
        raise SystemExit(
            f"FAIL {label}: observed {observed:.8g}, expected "
            f"[{lower:.8g}, {upper:.8g}]"
        )


def weighted_condition_mean(
    conditions: list[dict[str, Any]],
    select: Callable[[dict[str, Any]], float],
) -> float:
    total = sum(int(row["examples"]) for row in conditions)
    return sum(select(row) * int(row["examples"]) for row in conditions) / total


def verify_arm(arm_root: Path, result: dict[str, Any]) -> None:
    config = result["config"]
    expected = {
        "slots": 1_024,
        "reads": 16,
        "writes": 16,
        "memory_heads": 1,
        "seed": 0,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise SystemExit(f"FAIL {arm_root.name} has {name}={config.get(name)}")
    causality = load_json(arm_root / "CAUSALITY.json")
    if causality["maximum_prefix_difference"] != 0:
        raise SystemExit(f"FAIL prefix causality is not exact: {arm_root}")
    allocator = load_json(arm_root / "ALLOCATOR_VALIDATION.json")
    if (
        allocator["dense_packed_read_maximum_difference"] != 0
        or allocator["dense_packed_state_maximum_difference"] != 0
    ):
        raise SystemExit(f"FAIL copy-on-write state is not exact: {arm_root}")


def load_expected(path: Path) -> dict[float, dict[str, str]]:
    with path.open(newline="") as handle:
        return {
            float(row["occupancy_price"]): row
            for row in csv.DictReader(handle)
        }


def check_recall(output_root: Path, measurement_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for arm in RECALL_ARMS:
        arm_root = output_root / "recall" / arm
        result = load_json(arm_root / "result.json")
        verify_arm(arm_root, result)
        config = result["config"]
        aggregate = result["final_test"]["aggregate"]
        conditions = result["final_test"]["by_condition"]
        private_rows = weighted_condition_mean(
            conditions,
            lambda row: float(
                row["copy_on_write"]["unique_mutable_rows_final"]["mean"]
            ),
        )
        overlay_bytes = weighted_condition_mean(
            conditions,
            lambda row: float(
                row["copy_on_write"]["mutable_overlay_bytes_final"]["mean"]
            ),
        )
        logical_rows = config["layout"].count("B") * config["slots"]
        dense_bytes = result["parameter_accounting"][
            "logical_mutable_state_bytes_fp32_per_sequence"
        ]
        allocator_map_bytes = result["parameter_accounting"][
            "allocator_map_bytes_per_sequence"
        ]
        value_plus_map_bytes = overlay_bytes + allocator_map_bytes
        rows.append(
            {
                "occupancy_price": config["occupancy_price"],
                "query_loss": aggregate["query_loss"],
                "query_accuracy": aggregate["query_accuracy"],
                "exact_set_accuracy": aggregate["exact_set_accuracy"],
                "exact_group_accuracy": aggregate["exact_group_accuracy"],
                "mean_private_rows": private_rows,
                "logical_rows": logical_rows,
                "active_fraction": private_rows / logical_rows,
                "overlay_mib": overlay_bytes / (1 << 20),
                "allocator_map_mib": allocator_map_bytes / (1 << 20),
                "value_plus_map_mib": value_plus_map_bytes / (1 << 20),
                "dense_to_overlay_compression": dense_bytes / overlay_bytes,
                "dense_to_value_plus_map_compression": (
                    dense_bytes / value_plus_map_bytes
                ),
                "dense_state_savings": 1 - overlay_bytes / dense_bytes,
                "value_plus_map_savings": 1 - value_plus_map_bytes / dense_bytes,
                "row_reduction_vs_unpriced": 0.0,
            }
        )
    unpriced_rows = float(rows[0]["mean_private_rows"])
    for row in rows:
        row["row_reduction_vs_unpriced"] = (
            1 - float(row["mean_private_rows"]) / unpriced_rows
        )
    write_csv(measurement_root / "recall_curve.csv", rows)

    expected = load_expected(ROOT / "data/recall_curve.csv")
    for row in rows:
        price = float(row["occupancy_price"])
        reference = expected[price]
        require_close(
            f"λ={price:g} recall query loss",
            float(row["query_loss"]),
            float(reference["query_loss"]),
            0.03,
        )
        require_close(
            f"λ={price:g} recall exact-set accuracy",
            float(row["exact_set_accuracy"]),
            float(reference["exact_set_accuracy"]),
            0.03,
        )
        reference_rows = float(reference["mean_private_rows"])
        require_close(
            f"λ={price:g} recall private rows",
            float(row["mean_private_rows"]),
            reference_rows,
            max(25.0, reference_rows * 0.05),
        )
    indexed = {float(row["occupancy_price"]): row for row in rows}
    ordered = [indexed[price] for price in sorted(indexed)]
    if any(
        float(right["mean_private_rows"]) >= float(left["mean_private_rows"])
        for left, right in zip(ordered, ordered[1:])
    ):
        raise SystemExit("FAIL stronger recall prices did not reduce private rows")
    base = indexed[0.0]
    moderate = indexed[0.024]
    strong = indexed[0.12]
    require_between(
        "moderate-price recall row reduction",
        float(moderate["row_reduction_vs_unpriced"]),
        0.35,
        0.60,
    )
    require_between(
        "moderate-price recall loss cost",
        float(moderate["query_loss"]) - float(base["query_loss"]),
        -0.03,
        0.03,
    )
    require_between(
        "moderate-price recall exact-set change",
        float(moderate["exact_set_accuracy"])
        - float(base["exact_set_accuracy"]),
        -0.01,
        0.05,
    )
    if float(strong["mean_private_rows"]) >= float(moderate["mean_private_rows"]):
        raise SystemExit("FAIL the stronger price did not reduce private rows")
    print(f"PASS recall; fresh measurements: {measurement_root}", flush=True)


def check_language(output_root: Path, measurement_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for arm in LANGUAGE_ARMS:
        arm_root = output_root / "language" / arm
        result = load_json(arm_root / "result.json")
        verify_arm(arm_root, result)
        config = result["config"]
        trained = result["trained_occupancy"]
        prefix = next(
            row for row in trained["prefixes"] if int(row["prefix_tokens"]) == 2_048
        )
        private_rows = float(prefix["unique_rows"]["mean"])
        overlay_bytes = private_rows * (config["width"] + 1) * 4
        dense_bytes = result["parameter_accounting"][
            "logical_mutable_state_bytes_fp32_per_sequence"
        ]
        allocator_map_bytes = result["parameter_accounting"][
            "allocator_map_bytes_per_sequence"
        ]
        value_plus_map_bytes = overlay_bytes + allocator_map_bytes
        logical_rows = int(trained["logical_rows_per_sequence"])
        summary = result["curve_summary"]["complete_task_loss"]
        rows.append(
            {
                "occupancy_price": config["occupancy_price"],
                "test_nll": result["final_test"]["loss"],
                "test_perplexity": result["final_test"]["perplexity"],
                "validation_nll": result["final_validation"]["loss"],
                "curve_mean_task_loss": summary["mean"],
                "curve_auc_task_loss": summary["normalized_auc"],
                "mean_private_rows": private_rows,
                "prefix_tokens": prefix["prefix_tokens"],
                "logical_rows": logical_rows,
                "active_fraction": private_rows / logical_rows,
                "overlay_mib": overlay_bytes / (1 << 20),
                "allocator_map_mib": allocator_map_bytes / (1 << 20),
                "value_plus_map_mib": value_plus_map_bytes / (1 << 20),
                "dense_to_overlay_compression": dense_bytes / overlay_bytes,
                "dense_to_value_plus_map_compression": (
                    dense_bytes / value_plus_map_bytes
                ),
                "dense_state_savings": 1 - overlay_bytes / dense_bytes,
                "value_plus_map_savings": 1 - value_plus_map_bytes / dense_bytes,
                "row_reduction_vs_unpriced": 0.0,
                "accelerator": result["device"],
                "optimizer_seconds": result["optimizer_seconds"],
                "training_tokens_per_second": result[
                    "optimizer_tokens_per_second"
                ],
                "peak_gpu_allocation_bytes": result["peak_gpu_memory_bytes"],
            }
        )
    unpriced_rows = float(rows[0]["mean_private_rows"])
    for row in rows:
        row["row_reduction_vs_unpriced"] = (
            1 - float(row["mean_private_rows"]) / unpriced_rows
        )
    write_csv(measurement_root / "language_curve.csv", rows)

    expected = load_expected(ROOT / "data/language_curve.csv")
    for row in rows:
        price = float(row["occupancy_price"])
        reference = expected[price]
        require_close(
            f"λ={price:g} WikiText test NLL",
            float(row["test_nll"]),
            float(reference["test_nll"]),
            0.03,
        )
        reference_rows = float(reference["mean_private_rows"])
        require_close(
            f"λ={price:g} WikiText private rows",
            float(row["mean_private_rows"]),
            reference_rows,
            max(25.0, reference_rows * 0.05),
        )
    indexed = {float(row["occupancy_price"]): row for row in rows}
    base = indexed[0.0]
    priced = indexed[0.024]
    require_between(
        "priced WikiText row reduction",
        float(priced["row_reduction_vs_unpriced"]),
        0.15,
        0.60,
    )
    require_between(
        "priced WikiText NLL cost",
        float(priced["test_nll"]) - float(base["test_nll"]),
        -0.03,
        0.03,
    )
    print(f"PASS language; fresh measurements: {measurement_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("recall", "language", "all"))
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "runs/reproduction"
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    measurement_root = output_root / "measurements"
    suites = ("recall", "language") if args.suite == "all" else (args.suite,)
    for suite in suites:
        if suite == "recall":
            check_recall(output_root, measurement_root)
        else:
            check_language(output_root, measurement_root)


if __name__ == "__main__":
    main()
