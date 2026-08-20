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
    ("native_sdm_n1024_rw16_price000", 1_024, 0.0),
    ("native_sdm_n1024_rw16_price0024", 1_024, 0.024),
    ("native_sdm_n1024_rw16_price012", 1_024, 0.12),
    ("native_sdm_n1024_rw16_price0240", 1_024, 0.24),
    ("native_sdm_n1024_rw16_price0400", 1_024, 0.40),
    ("native_sdm_n2304_rw16_price000", 2_304, 0.0),
    ("native_sdm_n2304_rw16_price0054", 2_304, 0.054),
    ("native_sdm_n2304_rw16_price027", 2_304, 0.27),
)
LANGUAGE_ARMS = (
    ("native_sdm_rw16_price000", 0.0),
    ("native_sdm_rw16_price0024", 0.024),
    ("native_sdm_rw16_price012", 0.12),
    ("native_sdm_rw16_price0240", 0.24),
    ("native_sdm_rw16_price0400", 0.40),
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


def verify_arm(
    arm_root: Path,
    result: dict[str, Any],
    *,
    expected_seed: int,
    expected_slots: int,
    expected_price: float,
) -> None:
    config = result["config"]
    expected = {
        "slots": expected_slots,
        "reads": 16,
        "writes": 16,
        "memory_heads": 1,
        "seed": expected_seed,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise SystemExit(f"FAIL {arm_root.name} has {name}={config.get(name)}")
    if not math.isclose(
        float(config.get("occupancy_price", float("nan"))),
        expected_price,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            f"FAIL {arm_root.name} has occupancy_price="
            f"{config.get('occupancy_price')}"
        )
    causality = load_json(arm_root / "CAUSALITY.json")
    if causality["maximum_prefix_difference"] != 0:
        raise SystemExit(f"FAIL prefix causality is not exact: {arm_root}")
    allocator = load_json(arm_root / "ALLOCATOR_VALIDATION.json")
    if (
        allocator["dense_packed_read_maximum_difference"] != 0
        or allocator["dense_packed_state_maximum_difference"] != 0
    ):
        raise SystemExit(f"FAIL copy-on-write state is not exact: {arm_root}")


def load_expected(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_recall(output_root: Path, measurement_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for arm, capacity, price in RECALL_ARMS:
        arm_root = output_root / "recall" / arm
        result = load_json(arm_root / "result.json")
        verify_arm(
            arm_root,
            result,
            expected_seed=0,
            expected_slots=capacity,
            expected_price=price,
        )
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
                "capacity": capacity,
                "occupancy_price": config["occupancy_price"],
                "mean_private_row_coefficient": (
                    float(config["occupancy_price"]) / capacity
                ),
                "query_loss": aggregate["query_loss"],
                "query_accuracy": aggregate["query_accuracy"],
                "exact_set_accuracy": aggregate["exact_set_accuracy"],
                "exact_group_accuracy": aggregate["exact_group_accuracy"],
                "mean_private_rows": private_rows,
                "logical_rows": logical_rows,
                "active_fraction": private_rows / logical_rows,
                "overlay_bytes": overlay_bytes,
                "allocator_map_bytes": allocator_map_bytes,
                "value_plus_map_bytes": value_plus_map_bytes,
                "dense_state_bytes": dense_bytes,
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
                "private_rows_ratio_to_n1024": 1.0,
                "logical_capacity_ratio_to_n1024": capacity / 1_024,
                "shared_initial_memory_parameters": result["parameter_accounting"][
                    "learned_initial_state_parameters"
                ],
                "other_active_parameters": result["parameter_accounting"][
                    "active_parameters"
                ],
                "total_trainable_parameters": result["parameter_accounting"][
                    "total_trainable_parameters"
                ],
            }
        )
    unpriced_rows = {
        int(row["capacity"]): float(row["mean_private_rows"])
        for row in rows
        if float(row["occupancy_price"]) == 0.0
    }
    for row in rows:
        capacity = int(row["capacity"])
        row["row_reduction_vs_unpriced"] = (
            1 - float(row["mean_private_rows"]) / unpriced_rows[capacity]
        )
    indexed = {
        (int(row["capacity"]), float(row["occupancy_price"])): row
        for row in rows
    }
    for large_price, small_price in ((0.0, 0.0), (0.054, 0.024), (0.27, 0.12)):
        large = indexed[(2_304, large_price)]
        small = indexed[(1_024, small_price)]
        large["private_rows_ratio_to_n1024"] = (
            float(large["mean_private_rows"]) / float(small["mean_private_rows"])
        )
    price_rows = [row for row in rows if int(row["capacity"]) == 1_024]
    capacity_pairs = {
        (1_024, 0.0),
        (1_024, 0.024),
        (1_024, 0.12),
        (2_304, 0.0),
        (2_304, 0.054),
        (2_304, 0.27),
    }
    capacity_rows = [
        row
        for row in rows
        if (int(row["capacity"]), float(row["occupancy_price"]))
        in capacity_pairs
    ]
    write_csv(measurement_root / "recall_price_curve.csv", price_rows)
    write_csv(measurement_root / "recall_curve.csv", capacity_rows)

    expected: dict[tuple[int, float], dict[str, str]] = {}
    for relative in ("data/recall_price_curve.csv", "data/recall_curve.csv"):
        for row in load_expected(ROOT / relative):
            expected[(int(row["capacity"]), float(row["occupancy_price"]))] = row
    for row in rows:
        capacity = int(row["capacity"])
        price = float(row["occupancy_price"])
        reference = expected[(capacity, price)]
        require_close(
            f"N={capacity} λ={price:g} recall query loss",
            float(row["query_loss"]),
            float(reference["query_loss"]),
            0.03,
        )
        require_close(
            f"N={capacity} λ={price:g} recall exact-set accuracy",
            float(row["exact_set_accuracy"]),
            float(reference["exact_set_accuracy"]),
            0.03,
        )
        reference_rows = float(reference["mean_private_rows"])
        require_close(
            f"N={capacity} λ={price:g} recall private rows",
            float(row["mean_private_rows"]),
            reference_rows,
            max(25.0, reference_rows * 0.05),
        )
    fixed_capacity = [indexed[(1_024, price)] for price in (0.0, 0.024, 0.12, 0.24, 0.40)]
    if any(
        float(right["mean_private_rows"]) >= float(left["mean_private_rows"])
        for left, right in zip(fixed_capacity, fixed_capacity[1:])
    ):
        raise SystemExit("FAIL stronger recall prices did not reduce N=1,024 rows")
    for capacity, prices in (
        (1_024, (0.0, 0.024, 0.12)),
        (2_304, (0.0, 0.054, 0.27)),
    ):
        ordered = [indexed[(capacity, price)] for price in prices]
        if any(
            float(right["mean_private_rows"])
            >= float(left["mean_private_rows"])
            for left, right in zip(ordered, ordered[1:])
        ):
            raise SystemExit(
                f"FAIL stronger recall prices did not reduce N={capacity} rows"
            )
        base, moderate, strong = ordered
        require_between(
            f"N={capacity} moderate-price recall row reduction",
            float(moderate["row_reduction_vs_unpriced"]),
            0.25,
            0.65,
        )
        require_between(
            f"N={capacity} moderate-price recall loss cost",
            float(moderate["query_loss"]) - float(base["query_loss"]),
            -0.05,
            0.05,
        )
        require_between(
            f"N={capacity} moderate-price recall exact-set change",
            float(moderate["exact_set_accuracy"])
            - float(base["exact_set_accuracy"]),
            -0.05,
            0.05,
        )
        if float(strong["mean_private_rows"]) >= float(
            moderate["mean_private_rows"]
        ):
            raise SystemExit(
                f"FAIL N={capacity} stronger price did not reduce private rows"
            )
    for large_price in (0.0, 0.054, 0.27):
        ratio = float(indexed[(2_304, large_price)]["private_rows_ratio_to_n1024"])
        require_between(
            f"matched-price N=2,304 private-row ratio at λ={large_price:g}",
            ratio,
            0.5,
            2.0,
        )
    print(f"PASS recall; fresh measurements: {measurement_root}", flush=True)


def check_language(output_root: Path, measurement_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for arm, price in LANGUAGE_ARMS:
        arm_root = output_root / "language" / arm
        result = load_json(arm_root / "result.json")
        verify_arm(
            arm_root,
            result,
            expected_seed=0,
            expected_slots=1_024,
            expected_price=price,
        )
        config = result["config"]
        coverage = result["coverage"]
        if not (
            coverage.get("complete") is True
            and coverage.get("all_passes_exactly_once_without_replacement") is True
            and int(coverage.get("consumed_optimizer_steps", -1)) == 21_603
            and int(coverage.get("consumed_target_presentations", -1))
            == 353_941_347
        ):
            raise SystemExit(f"FAIL {arm} did not complete canonical WikiText coverage")
        validation = result["validation_by_pass"]
        if sorted(validation) != ["1", "2", "3"]:
            raise SystemExit(f"FAIL {arm} lacks the three validation passes")
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
        logical_rows = int(trained["logical_rows_per_request"])
        summary = result["curve_summary"]["complete_train_nll"]
        rows.append(
            {
                "occupancy_price": config["occupancy_price"],
                "validation_nll_pass1": validation["1"]["nll"],
                "validation_nll_pass2": validation["2"]["nll"],
                "validation_nll_pass3": validation["3"]["nll"],
                "test_nll": result["final_test"]["loss"],
                "test_perplexity": result["final_test"]["perplexity"],
                "test_bits_per_utf8_byte": result["final_test"][
                    "bits_per_utf8_byte"
                ],
                "test_next_token_accuracy": result["final_test"][
                    "next_token_accuracy"
                ],
                "curve_mean_train_nll": summary["mean"],
                "curve_auc_train_nll": summary["normalized_auc"],
                "mean_private_rows": private_rows,
                "mean_private_rows_per_bank": prefix[
                    "mean_private_rows_per_bank"
                ]["mean"],
                "prefix_tokens": prefix["prefix_tokens"],
                "logical_rows": logical_rows,
                "active_fraction": private_rows / logical_rows,
                "overlay_bytes": overlay_bytes,
                "allocator_map_bytes": allocator_map_bytes,
                "value_plus_map_bytes": value_plus_map_bytes,
                "dense_state_bytes": dense_bytes,
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
                "optimizer_targets_per_second": result[
                    "optimizer_targets_per_second"
                ],
                "peak_gpu_allocation_bytes": result["peak_gpu_memory_bytes"],
                "total_seconds": result["total_seconds"],
                "optimizer_steps": coverage["consumed_optimizer_steps"],
                "target_presentations": coverage[
                    "consumed_target_presentations"
                ],
                "passes": len(coverage["passes"]),
                "protocol_id": result["protocol_id"],
            }
        )
    unpriced_rows = float(rows[0]["mean_private_rows"])
    for row in rows:
        row["row_reduction_vs_unpriced"] = (
            1 - float(row["mean_private_rows"]) / unpriced_rows
        )
    write_csv(measurement_root / "language_curve.csv", rows)

    expected = {
        float(row["occupancy_price"]): row
        for row in load_expected(ROOT / "data/language_curve.csv")
    }
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
    ordered = [indexed[price] for price in (0.0, 0.024, 0.12, 0.24)]
    if any(
        float(right["mean_private_rows"]) >= float(left["mean_private_rows"])
        for left, right in zip(ordered, ordered[1:])
    ):
        raise SystemExit(
            "FAIL WikiText prices through λ=0.24 did not reduce private rows"
        )
    base = indexed[0.0]
    priced = indexed[0.24]
    overprice = indexed[0.4]
    require_between(
        "priced WikiText row reduction",
        float(priced["row_reduction_vs_unpriced"]),
        0.60,
        0.85,
    )
    require_between(
        "priced WikiText NLL cost",
        float(priced["test_nll"]) - float(base["test_nll"]),
        -0.03,
        0.03,
    )
    if float(overprice["mean_private_rows"]) <= float(priced["mean_private_rows"]):
        raise SystemExit("FAIL λ=0.40 did not reproduce the measured state boundary")
    if float(overprice["test_nll"]) <= float(priced["test_nll"]):
        raise SystemExit("FAIL λ=0.40 did not reproduce the measured quality boundary")
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
