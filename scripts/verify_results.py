#!/usr/bin/env python3
"""Verify the compact recorded measurements without rerunning training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROW_BYTES = (128 + 1) * 4


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL {message}")


def close(observed: float, expected: float, *, tolerance: float = 2e-7) -> bool:
    return math.isfinite(observed) and math.isclose(
        observed,
        expected,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_state_columns(
    table: list[dict[str, str]],
    label: str,
    *,
    grouped_by_capacity: bool,
) -> None:
    base_rows: dict[int, float] = {}
    for row in table:
        capacity = int(row.get("capacity", "1024"))
        group = capacity if grouped_by_capacity else 0
        if float(row["occupancy_price"]) == 0.0:
            base_rows[group] = float(row["mean_private_rows"])
    for row in table:
        capacity = int(row.get("capacity", "1024"))
        group = capacity if grouped_by_capacity else 0
        logical_rows = 7 * capacity
        map_bytes = logical_rows * 4
        dense_bytes = 7 * capacity * 128 * 4
        private_rows = float(row["mean_private_rows"])
        overlay_bytes = private_rows * ROW_BYTES
        value_plus_map_bytes = overlay_bytes + map_bytes
        expected = {
            "logical_rows": float(logical_rows),
            "active_fraction": private_rows / logical_rows,
            "overlay_bytes": overlay_bytes,
            "allocator_map_bytes": float(map_bytes),
            "value_plus_map_bytes": value_plus_map_bytes,
            "dense_state_bytes": float(dense_bytes),
            "overlay_mib": overlay_bytes / (1 << 20),
            "allocator_map_mib": map_bytes / (1 << 20),
            "value_plus_map_mib": value_plus_map_bytes / (1 << 20),
            "dense_to_overlay_compression": dense_bytes / overlay_bytes,
            "dense_to_value_plus_map_compression": dense_bytes / value_plus_map_bytes,
            "dense_state_savings": 1 - overlay_bytes / dense_bytes,
            "value_plus_map_savings": 1 - value_plus_map_bytes / dense_bytes,
            "row_reduction_vs_unpriced": 1 - private_rows / base_rows[group],
        }
        price = row["occupancy_price"]
        for name, value in expected.items():
            require(
                close(float(row[name]), value),
                f"{label} λ={price} has inconsistent {name}",
            )


def verify_recall() -> None:
    table = rows(ROOT / "data/recall_curve.csv")
    require(len(table) == 6, "recall table must contain exactly six arms")
    require(
        [
            (int(row["capacity"]), float(row["occupancy_price"]))
            for row in table
        ]
        == [
            (1_024, 0.0),
            (1_024, 0.024),
            (1_024, 0.12),
            (2_304, 0.0),
            (2_304, 0.054),
            (2_304, 0.27),
        ],
        "recall capacity-price pairs do not match the released matrix",
    )
    verify_state_columns(table, "recall", grouped_by_capacity=True)
    indexed = {
        (int(row["capacity"]), float(row["occupancy_price"])): row
        for row in table
    }
    for row in table:
        capacity = int(row["capacity"])
        price = float(row["occupancy_price"])
        require(
            close(
                float(row["mean_private_row_coefficient"]),
                price / capacity,
            ),
            f"recall N={capacity} λ={price:g} has inconsistent mean-row coefficient",
        )
        require(
            close(
                float(row["logical_capacity_ratio_to_n1024"]),
                capacity / 1_024,
            ),
            f"recall N={capacity} λ={price:g} has inconsistent capacity ratio",
        )
        shared = int(row["shared_initial_memory_parameters"])
        other = int(row["other_active_parameters"])
        total = int(row["total_trainable_parameters"])
        require(
            shared == 7 * capacity * 128,
            f"recall N={capacity} λ={price:g} has inconsistent shared memory",
        )
        require(
            total == shared + other,
            f"recall N={capacity} λ={price:g} has inconsistent parameters",
        )
    capacity_ratio = 2_304 / 1_024
    for large_price, small_price in ((0.0, 0.0), (0.054, 0.024), (0.27, 0.12)):
        large = indexed[(2_304, large_price)]
        small = indexed[(1_024, small_price)]
        expected_ratio = float(large["mean_private_rows"]) / float(
            small["mean_private_rows"]
        )
        empirical_exponent = math.log(expected_ratio) / math.log(capacity_ratio)
        require(
            close(float(large["private_rows_ratio_to_n1024"]), expected_ratio),
            f"recall N=2,304 λ={large_price:g} has inconsistent row ratio",
        )
        require(
            empirical_exponent <= 0.5,
            f"recall N=2,304 λ={large_price:g} exceeds the square-root-growth canary",
        )
        if large_price != 0.0:
            require(
                empirical_exponent <= 0.25,
                f"recall N=2,304 λ={large_price:g} exceeds the fourth-root reference",
            )
    for row in table:
        require(float(row["query_loss"]) > 0, "recall query loss is invalid")
        for name in (
            "query_accuracy",
            "exact_set_accuracy",
            "exact_group_accuracy",
        ):
            require(0 <= float(row[name]) <= 1, f"recall {name} is invalid")


def verify_recall_price_curve() -> None:
    table = rows(ROOT / "data/recall_price_curve.csv")
    require(len(table) == 5, "recall price table must contain exactly five arms")
    require(
        [float(row["occupancy_price"]) for row in table]
        == [0.0, 0.024, 0.12, 0.24, 0.4],
        "recall prices do not match the WikiText price grid",
    )
    require(
        all(int(row["capacity"]) == 1_024 for row in table),
        "recall price table is not fixed at N=1,024",
    )
    verify_state_columns(table, "recall price", grouped_by_capacity=False)
    private_rows = [float(row["mean_private_rows"]) for row in table]
    require(
        all(right < left for left, right in zip(private_rows, private_rows[1:])),
        "recall private rows do not decrease across the price grid",
    )
    for row in table:
        require(float(row["query_loss"]) > 0, "recall query loss is invalid")
        require(
            0 <= float(row["exact_set_accuracy"]) <= 1,
            "recall exact-set accuracy is invalid",
        )

    capacity_rows = {
        (int(row["capacity"]), float(row["occupancy_price"])): row
        for row in rows(ROOT / "data/recall_curve.csv")
    }
    for row in table[:3]:
        key = (1_024, float(row["occupancy_price"]))
        require(
            row == capacity_rows[key],
            f"recall anchor N=1,024 λ={row['occupancy_price']} is inconsistent",
        )


def verify_language() -> None:
    table = rows(ROOT / "data/language_curve.csv")
    require(len(table) == 5, "language table must contain exactly five arms")
    require(
        [float(row["occupancy_price"]) for row in table]
        == [0.0, 0.024, 0.12, 0.24, 0.4],
        "language prices do not match the released curve",
    )
    verify_state_columns(table, "language", grouped_by_capacity=False)
    for row in table:
        nll = float(row["test_nll"])
        require(nll > 0, "WikiText test NLL is invalid")
        require(
            close(float(row["test_perplexity"]), math.exp(nll), tolerance=2e-8),
            f"WikiText λ={row['occupancy_price']} perplexity is inconsistent",
        )
        require(int(row["prefix_tokens"]) == 2_048, "WikiText prefix is not 2,048")
        require(
            close(
                float(row["mean_private_rows_per_bank"]),
                float(row["mean_private_rows"]) / 7,
            ),
            f"WikiText λ={row['occupancy_price']} has inconsistent per-bank rows",
        )
        require(int(row["optimizer_steps"]) == 21_603, "WikiText steps changed")
        require(
            int(row["target_presentations"]) == 353_941_347,
            "WikiText target coverage changed",
        )
        require(int(row["passes"]) == 3, "WikiText pass count changed")
        require(
            row["protocol_id"]
            == "wikitext103-gpt2-causal-t2048-coverage-v1",
            "WikiText protocol identity changed",
        )
        require(
            0 <= float(row["test_next_token_accuracy"]) <= 1,
            "WikiText next-token accuracy is invalid",
        )
        passes = [float(row[f"validation_nll_pass{index}"]) for index in (1, 2, 3)]
        require(
            all(right < left for left, right in zip(passes, passes[1:])),
            f"WikiText λ={row['occupancy_price']} validation did not improve by pass",
        )


def verify_provenance() -> None:
    provenance = json.loads((ROOT / "provenance.json").read_text())
    require(provenance["schema_version"] == 7, "unexpected provenance schema")
    require(
        provenance["implementation"]["shape"]
        == {
            "layout": "BBBBBBBA",
            "width": 128,
            "attention_heads": 4,
            "memory_heads": 1,
            "slots_per_memory_layer": [1_024, 2_304],
            "reads": 16,
            "writes": 16,
        },
        "provenance shape is not the balanced-write experiment",
    )
    require(
        provenance["data"]["wikitext_manifest_sha256"]
        == "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6",
        "WikiText manifest provenance changed",
    )
    require(
        provenance["data"]["wikitext_payload_sha256"]
        == "f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e",
        "WikiText payload provenance changed",
    )
    require(
        provenance["data"]["recall_manifest_sha256"]
        == "b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800",
        "recall manifest provenance changed",
    )
    expected_arms = {"recall": 8, "language": 5}
    for name, count in expected_arms.items():
        campaign = provenance["campaigns"][name]
        require(len(campaign["results"]) == count, f"unexpected {name} arm count")
        runs = campaign.get("runs", [])
        require(bool(runs), f"{name} provenance has no campaign runs")
        run_arms = {
            arm
            for run in runs
            for arm in run.get("arms", [])
        }
        require(
            run_arms == set(campaign["results"]),
            f"{name} run arms do not match its result records",
        )
        for run in runs:
            require(
                close(
                    float(run["billed_cost_usd"]),
                    float(run["billed_gpu_hours"])
                    * float(run["hourly_rate_usd"]),
                ),
                f"{name} run cost arithmetic is inconsistent",
            )
        require(
            close(
                float(campaign["billed_gpu_hours"]),
                sum(float(run["billed_gpu_hours"]) for run in runs),
            ),
            f"{name} aggregate GPU-hours are inconsistent",
        )
        require(
            close(
                float(campaign["billed_cost_usd"]),
                sum(float(run["billed_cost_usd"]) for run in runs),
            ),
            f"{name} aggregate cost is inconsistent",
        )
    for section in (
        "released_documents",
        "released_sources",
        "included_measurements",
        "included_figures",
    ):
        entries = provenance.get(section)
        require(bool(entries), f"provenance has no {section}")
        for relative, expected in entries.items():
            path = ROOT / relative
            require(path.is_file(), f"provenance target is missing: {relative}")
            require(digest(path) == expected, f"hash mismatch: {relative}")


def main() -> None:
    verify_recall()
    verify_recall_price_curve()
    verify_language()
    verify_provenance()
    print("PASS recorded balanced-write results and provenance", flush=True)


if __name__ == "__main__":
    main()
