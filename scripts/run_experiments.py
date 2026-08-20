#!/usr/bin/env python3
"""Run the released recall and WikiText usage-pricing experiments."""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
RECALL_HASH = "b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800"
LANGUAGE_MANIFEST_HASH = (
    "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6"
)
LANGUAGE_PAYLOAD_HASH = (
    "f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e"
)
LANGUAGE_DOUBLE_BUILD_HASH = (
    "888d5d27b8086223d48b2347364428947603ab4400a0b7c594d9b350baca7aa0"
)
LANGUAGE_LOADER_HASH = (
    "32d204f4f4ad64b9cd1c35a1b1e84680effb515269fa830be0093eca6416692b"
)
LANGUAGE_DESIGN_HASH = (
    "1a98779e6d5717c076503f3300b9c2f9d771f78a36b990e9c94d0ae194332293"
)
LANGUAGE_PROTOCOL = "wikitext103-gpt2-causal-t2048-coverage-v1"
LANGUAGE_CAMPAIGN = "esdm-wt103-n1024-rw16-price-s0-v1"

RECALL_ARMS = (
    ("native_sdm_n1024_rw16_price000", 1_024, 0.0, 0, 1_691_164),
    ("native_sdm_n1024_rw16_price0024", 1_024, 0.024, 0, 1_691_164),
    ("native_sdm_n1024_rw16_price012", 1_024, 0.12, 0, 1_691_164),
    ("native_sdm_n1024_rw16_price0240", 1_024, 0.24, 0, 1_691_164),
    ("native_sdm_n1024_rw16_price0400", 1_024, 0.40, 0, 1_691_164),
    ("native_sdm_n2304_rw16_price000", 2_304, 0.0, 0, 1_748_956),
    ("native_sdm_n2304_rw16_price0054", 2_304, 0.054, 0, 1_748_956),
    ("native_sdm_n2304_rw16_price027", 2_304, 0.27, 0, 1_748_956),
)
LANGUAGE_ARMS = (
    ("native_sdm_rw16_price000", 1_024, 0.0, 14_712_348),
    ("native_sdm_rw16_price0024", 1_024, 0.024, 14_712_348),
    ("native_sdm_rw16_price012", 1_024, 0.12, 14_712_348),
    ("native_sdm_rw16_price0240", 1_024, 0.24, 14_712_348),
    ("native_sdm_rw16_price0400", 1_024, 0.40, 14_712_348),
)


def recall_command(
    manifest: Path,
    output: Path,
    arm: tuple[str, int, float, int, int],
) -> list[str]:
    name, slots, price, seed, parameters = arm
    return [
        sys.executable, "-m", "benchmarks.train_recall_cuda",
        "--manifest", str(manifest), "--expected-manifest-sha256", RECALL_HASH,
        "--output-dir", str(output / name), "--arm", name,
        "--layout", "BBBBBBBA", "--layers", "8",
        "--steps", "30000", "--schedule-steps", "30000",
        "--batch-size", "32", "--micro-batch-size", "32",
        "--eval-batch-size", "32", "--checkpoint-eval-examples", "256",
        "--final-eval-examples", "2048", "--width", "128", "--heads", "4",
        "--slots", str(slots), "--reads", "16", "--writes", "16",
        "--memory-heads", "1", "--mlp-expansion", "4",
        "--learning-rate", "0.0003", "--weight-decay", "0.01",
        "--warmup-steps", "100", "--seed", str(seed),
        "--activation-dtype", "bfloat16", "--occupancy-price", str(price),
        "--expected-active-parameters", str(parameters),
        "--recovery-checkpoint-interval", "5000", "--no-save-checkpoint",
    ]


def language_command(manifest: Path, output: Path, arm: tuple[str, int, float, int]) -> list[str]:
    name, slots, price, parameters = arm
    return [
        sys.executable, "-m", "benchmarks.train_wikitext_coverage_cuda",
        "--manifest", str(manifest),
        "--expected-manifest-sha256", LANGUAGE_MANIFEST_HASH,
        "--expected-data-payload-sha256", LANGUAGE_PAYLOAD_HASH,
        "--expected-double-build-sha256", LANGUAGE_DOUBLE_BUILD_HASH,
        "--expected-loader-sha256", LANGUAGE_LOADER_HASH,
        "--output-dir", str(output / name),
        "--campaign", LANGUAGE_CAMPAIGN,
        "--campaign-design-sha256", LANGUAGE_DESIGN_HASH,
        "--protocol-id", LANGUAGE_PROTOCOL,
        "--arm", name, "--layout", "BBBBBBBA", "--passes", "3",
        "--steps", "21603", "--schedule-steps", "21603",
        "--batch-size", "8", "--micro-batch-size", "1",
        "--heartbeat-steps", "100", "--serving-route-examples", "64",
        "--serving-route-lengths", "16,64,256,1024,2048",
        "--width", "128", "--heads", "4", "--slots", str(slots),
        "--reads", "16", "--writes", "16", "--memory-heads", "1",
        "--mlp-expansion", "4", "--learning-rate", "0.0003",
        "--weight-decay", "0.01", "--warmup-steps", "540", "--seed", "0",
        "--stream-seed", "20260818",
        "--activation-dtype", "bfloat16", "--occupancy-price", str(price),
        "--expected-active-parameters", str(parameters),
        "--recovery-checkpoint-interval", "2400",
    ]


def run_parallel(commands: list[tuple[str, list[str]]], gpus: list[str]) -> None:
    pending = deque(commands)
    active: dict[str, tuple[str, subprocess.Popen]] = {}
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            name, command = pending.popleft()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["ELASTIC_SDM_REQUIRE_RELEASED_FUSED"] = "1"
            print(f"START {name} on GPU {gpu}", flush=True)
            active[gpu] = (name, subprocess.Popen(command, cwd=ROOT, env=environment))
        time.sleep(1)
        for gpu, (name, process) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[gpu]
            if code:
                for _, other in active.values():
                    other.terminate()
                raise SystemExit(f"{name} failed with exit code {code}")
            print(f"COMPLETE {name}", flush=True)


def parse_gpus(value: str | None) -> list[str]:
    if value:
        result = [row.strip() for row in value.split(",") if row.strip()]
    else:
        result = [str(index) for index in range(torch.cuda.device_count())]
    if not result:
        raise SystemExit("A CUDA GPU is required; pass --gpus with visible device IDs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("recall", "language", "all"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "runs/data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/reproduction")
    parser.add_argument("--gpus", help="comma-separated physical GPU IDs; one arm runs per GPU")
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_released_sdm_extensions.py")],
        cwd=ROOT,
        check=True,
    )
    suites = ("recall", "language") if args.suite == "all" else (args.suite,)
    print(
        "Complete balanced-write schedules: five matched seed-0 recall and "
        "WikiText prices at N=1,024, plus three seed-0 N=2,304 recall arms; "
        "every arm uses H=1 and R=W=16.",
        flush=True,
    )
    for suite in suites:
        output = args.output_root.resolve() / suite
        output.mkdir(parents=True, exist_ok=True)
        if suite == "recall":
            manifest = args.data_root.resolve() / "adaptive_recall_seed102337_v1/manifest.json"
            arms = RECALL_ARMS
            build = recall_command
        else:
            manifest = (
                args.data_root.resolve()
                / "wikitext103_gpt2_causal_t2048_coverage_v1/manifest.json"
            )
            arms = LANGUAGE_ARMS
            build = language_command
        if not manifest.exists():
            raise SystemExit(f"missing prepared data: {manifest}")
        commands = [(f"{suite}/{arm[0]}", build(manifest, output, arm)) for arm in arms]
        run_parallel(commands, gpus)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_results.py"),
                suite,
                "--output-root",
                str(args.output_root.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
