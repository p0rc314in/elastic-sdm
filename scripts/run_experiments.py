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

RECALL_ARMS = (
    ("native_sdm_rw16_price000", 1_024, 0.0, 1_691_164),
    ("native_sdm_rw16_price0012", 1_024, 0.012, 1_691_164),
    ("native_sdm_rw16_price0024", 1_024, 0.024, 1_691_164),
    ("native_sdm_rw16_price0040", 1_024, 0.04, 1_691_164),
    ("native_sdm_rw16_price012", 1_024, 0.12, 1_691_164),
    ("native_sdm_rw16_price0240", 1_024, 0.24, 1_691_164),
    ("native_sdm_rw16_price0400", 1_024, 0.40, 1_691_164),
)
LANGUAGE_ARMS = (
    ("native_sdm_rw16_price000", 1_024, 0.0, 14_712_348),
    ("native_sdm_rw16_price0024", 1_024, 0.024, 14_712_348),
)


def recall_command(manifest: Path, output: Path, arm: tuple[str, int, float, int]) -> list[str]:
    name, slots, price, parameters = arm
    return [
        sys.executable, "-m", "benchmarks.train_recall_cuda",
        "--manifest", str(manifest), "--expected-manifest-sha256", RECALL_HASH,
        "--output-dir", str(output / name), "--arm", name,
        "--layout", "BBBBBBBA", "--layers", "8",
        "--steps", "30000", "--schedule-steps", "30000",
        "--batch-size", "32", "--micro-batch-size", "8",
        "--eval-batch-size", "8", "--checkpoint-eval-examples", "256",
        "--final-eval-examples", "2048", "--width", "128", "--heads", "4",
        "--slots", str(slots), "--reads", "16", "--writes", "16",
        "--memory-heads", "1", "--mlp-expansion", "4",
        "--learning-rate", "0.0003", "--weight-decay", "0.01",
        "--warmup-steps", "100", "--seed", "0",
        "--activation-dtype", "bfloat16", "--occupancy-price", str(price),
        "--expected-active-parameters", str(parameters),
        "--recovery-checkpoint-interval", "5000", "--no-save-checkpoint",
    ]


def language_command(manifest: Path, output: Path, arm: tuple[str, int, float, int]) -> list[str]:
    name, slots, price, parameters = arm
    return [
        sys.executable, "-m", "benchmarks.train_wikitext_cuda",
        "--manifest", str(manifest), "--output-dir", str(output / name),
        "--arm", name, "--layout", "BBBBBBBA",
        "--steps", "4000", "--schedule-steps", "4000",
        "--batch-size", "8", "--micro-batch-size", "1",
        "--eval-batch-size", "1", "--checkpoint-eval-examples", "32",
        "--final-eval-examples", "128", "--serving-route-examples", "64",
        "--serving-route-lengths", "16,64,256,1024,2048",
        "--width", "128", "--heads", "4", "--slots", str(slots),
        "--reads", "16", "--writes", "16", "--memory-heads", "1",
        "--mlp-expansion", "4", "--learning-rate", "0.0003",
        "--weight-decay", "0.01", "--warmup-steps", "100", "--seed", "0",
        "--activation-dtype", "bfloat16", "--occupancy-price", str(price),
        "--expected-active-parameters", str(parameters),
        "--recovery-checkpoint-interval", "1000", "--no-save-checkpoint",
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
            output = Path(process.args[process.args.index("--output-dir") + 1])
            (output / "recovery_checkpoint.pt").unlink(missing_ok=True)
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
    suites = ("recall", "language") if args.suite == "all" else (args.suite,)
    print(
        "Complete seed-0 balanced-write schedules: seven recall arms and "
        "two WikiText arms at H=1, N=1,024, R=W=16.",
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
            manifest = args.data_root.resolve() / "wikitext103_gpt2_causal_2048/manifest.json"
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
