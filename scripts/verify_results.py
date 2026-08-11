#!/usr/bin/env python3
"""Verify the compact recorded measurements without rerunning training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGICAL_ROWS = 7 * 1_024
ROW_BYTES = (128 + 1) * 4
MAP_BYTES = LOGICAL_ROWS * 4
DENSE_BYTES = 7 * 1_024 * 128 * 4


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


def verify_state_columns(table: list[dict[str, str]], label: str) -> None:
    base_rows = float(table[0]["mean_private_rows"])
    for row in table:
        private_rows = float(row["mean_private_rows"])
        overlay_bytes = private_rows * ROW_BYTES
        value_plus_map_bytes = overlay_bytes + MAP_BYTES
        expected = {
            "logical_rows": float(LOGICAL_ROWS),
            "active_fraction": private_rows / LOGICAL_ROWS,
            "overlay_mib": overlay_bytes / (1 << 20),
            "allocator_map_mib": MAP_BYTES / (1 << 20),
            "value_plus_map_mib": value_plus_map_bytes / (1 << 20),
            "dense_to_overlay_compression": DENSE_BYTES / overlay_bytes,
            "dense_to_value_plus_map_compression": DENSE_BYTES / value_plus_map_bytes,
            "dense_state_savings": 1 - overlay_bytes / DENSE_BYTES,
            "value_plus_map_savings": 1 - value_plus_map_bytes / DENSE_BYTES,
            "row_reduction_vs_unpriced": 1 - private_rows / base_rows,
        }
        price = row["occupancy_price"]
        for name, value in expected.items():
            require(
                close(float(row[name]), value),
                f"{label} λ={price} has inconsistent {name}",
            )


def verify_recall() -> None:
    table = rows(ROOT / "data/recall_curve.csv")
    require(len(table) == 7, "recall table must contain exactly seven arms")
    require(
        [float(row["occupancy_price"]) for row in table]
        == [0.0, 0.012, 0.024, 0.04, 0.12, 0.24, 0.4],
        "recall prices do not match the released sweep",
    )
    verify_state_columns(table, "recall")
    for row in table:
        require(float(row["query_loss"]) > 0, "recall query loss is invalid")
        for name in (
            "query_accuracy",
            "exact_set_accuracy",
            "exact_group_accuracy",
        ):
            require(0 <= float(row[name]) <= 1, f"recall {name} is invalid")


def verify_language() -> None:
    table = rows(ROOT / "data/language_curve.csv")
    require(len(table) == 2, "language table must contain exactly two arms")
    require(
        [float(row["occupancy_price"]) for row in table] == [0.0, 0.024],
        "language prices do not match the released pair",
    )
    verify_state_columns(table, "language")
    for row in table:
        nll = float(row["test_nll"])
        require(nll > 0, "WikiText test NLL is invalid")
        require(
            close(float(row["test_perplexity"]), math.exp(nll), tolerance=2e-8),
            f"WikiText λ={row['occupancy_price']} perplexity is inconsistent",
        )
        require(int(row["prefix_tokens"]) == 2_048, "WikiText prefix is not 2,048")


def verify_provenance() -> None:
    provenance = json.loads((ROOT / "provenance.json").read_text())
    require(provenance["schema_version"] == 4, "unexpected provenance schema")
    require(
        provenance["implementation"]["shape"]
        == {
            "layout": "BBBBBBBA",
            "width": 128,
            "attention_heads": 4,
            "memory_heads": 1,
            "slots_per_memory_layer": 1_024,
            "reads": 16,
            "writes": 16,
            "seed": 0,
        },
        "provenance shape is not the balanced-write experiment",
    )
    require(
        provenance["data"]["wikitext_manifest_sha256"]
        == "b1bb41b7bc8f9c1fe4bb22820e6d242d67d4cc143f3763c371ff6ea6e6fd987d",
        "WikiText manifest provenance changed",
    )
    require(
        provenance["data"]["recall_manifest_sha256"]
        == "b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800",
        "recall manifest provenance changed",
    )
    expected_arms = {"recall": 7, "language": 2}
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
    verify_language()
    verify_provenance()
    print("PASS recorded balanced-write results and provenance", flush=True)


if __name__ == "__main__":
    main()
