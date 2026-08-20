#!/usr/bin/env python3
"""Train one native Elastic SDM arm on canonical WikiText-103 coverage."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.wikitext_support import (
    MODEL_IDENTITY,
    VALUE_ELEMENT_BYTES,
    TrainConfig,
    atomic_json,
    atomic_torch_save,
    curve_summary,
    learning_rate,
    native_routing,
    parameter_accounting,
    profiler_evidence,
    route_occupancy_counts,
    set_seed,
    sha256_file,
    to_device,
    verify_allocator,
    verify_native_sdm,
    verify_prefix_causality,
)
from benchmarks.wikitext103_coverage import (
    CONTEXT_LENGTH,
    EFFECTIVE_BATCH_RECORDS,
    EVALUATION_STRIDE,
    OPTIMIZER_STEPS_PER_PASS,
    PROTOCOL_ID,
    STREAM_SEED,
    VOCAB_SIZE,
    CanonicalWikiTextData,
)
from elastic_sdm.model import LanguageModel
from elastic_sdm.sdm import aggregate_sdm_copy_on_write_accounting


CANONICAL_PASSES = 3
CANONICAL_STEPS = CANONICAL_PASSES * OPTIMIZER_STEPS_PER_PASS
CANONICAL_WARMUP_STEPS = round(CANONICAL_STEPS * 0.025)
RECOVERY_KIND = "elastic_sdm_canonical_wikitext_recovery_v1"
SLOT_ID_BYTES = 4


def bool_to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.bool_, copy=False)).to(
        device, non_blocking=True
    )


def describe(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


@torch.no_grad()
def evaluate(
    model: LanguageModel,
    data: CanonicalWikiTextData,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    """Score every declared target in one canonical held-out split."""

    model.eval()
    total_nll = 0.0
    correct_targets = 0
    scored_targets = 0
    scored_bytes = 0
    windows = 0
    started = time.perf_counter()
    for batch in data.evaluation_batches(split):
        values = to_device(batch.tokens, device)
        mask = bool_to_device(batch.target_mask, device)
        inputs, labels = values[:, :-1], values[:, 1:]
        with torch.autocast("cuda", dtype=model.activation_dtype):
            logits = model(inputs)
        losses = F.cross_entropy(
            logits.float().flatten(0, 1),
            labels.flatten(),
            reduction="none",
        ).view_as(labels)
        total_nll += float(losses[mask].sum())
        correct_targets += int(logits.argmax(dim=-1).eq(labels)[mask].sum())
        scored_targets += batch.target_count
        scored_bytes += batch.scored_bytes
        windows += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    declared = data.manifest["evaluation"]["splits"][split]
    if scored_targets != int(declared["scored_targets"]):
        raise ValueError(f"{split} evaluation target coverage changed")
    if scored_bytes != int(declared["scored_bytes"]):
        raise ValueError(f"{split} evaluation byte coverage changed")
    nll = total_nll / scored_targets
    return {
        "split": split,
        "protocol_id": PROTOCOL_ID,
        "context_length": CONTEXT_LENGTH,
        "stride": EVALUATION_STRIDE,
        "windows": windows,
        "total_nll": total_nll,
        "nll": nll,
        "loss": nll,
        "perplexity": math.exp(nll),
        "bits_per_utf8_byte": total_nll / (math.log(2.0) * scored_bytes),
        "next_token_accuracy": correct_targets / scored_targets,
        "token_accuracy": correct_targets / scored_targets,
        "correct_targets": correct_targets,
        "scored_targets": scored_targets,
        "tokens": scored_targets,
        "scored_bytes": scored_bytes,
        "seconds": elapsed,
        "scored_targets_per_second": scored_targets / elapsed,
    }


def canonical_validation_inputs(
    data: CanonicalWikiTextData,
    *,
    examples: int,
    prefix: int,
) -> list[np.ndarray]:
    """Select fixed validation windows that contain a complete prefix."""

    selected: list[np.ndarray] = []
    for batch in data.evaluation_batches("validation"):
        if batch.tokens.shape[1] - 1 < prefix:
            continue
        selected.append(np.asarray(batch.tokens[:, : prefix + 1]))
        if len(selected) == examples:
            break
    if len(selected) != examples:
        raise ValueError(
            f"only {len(selected)} validation windows contain {prefix} input tokens"
        )
    return selected


@torch.no_grad()
def export_trained_occupancy(
    model: LanguageModel,
    data: CanonicalWikiTextData,
    device: torch.device,
    *,
    examples: int,
    prefix_lengths: tuple[int, ...],
    output: Path,
) -> dict[str, Any]:
    """Export private-row counts on fixed full-context validation windows."""

    if not prefix_lengths or min(prefix_lengths) <= 0:
        raise ValueError("occupancy prefixes must be positive")
    maximum_prefix = max(prefix_lengths)
    if maximum_prefix > CONTEXT_LENGTH:
        raise ValueError("occupancy prefix exceeds the canonical context")
    model.eval()
    all_routes: list[torch.Tensor] = []
    for batch in canonical_validation_inputs(
        data,
        examples=examples,
        prefix=maximum_prefix,
    ):
        values = to_device(batch, device)
        with torch.autocast("cuda", dtype=model.activation_dtype):
            _, routing = model(values[:, :-1], return_routing=True)
        layers = [
            row.write_indices.to(torch.int16).cpu()
            for row in native_routing(routing)
        ]
        all_routes.append(torch.stack(layers, dim=1))
    routes = torch.cat(all_routes, dim=0)
    route_path = output / "serving_write_routes.pt"
    atomic_torch_save(
        route_path,
        {
            "schema": "native-sdm-serving-write-routes-v2",
            "protocol_id": PROTOCOL_ID,
            "split": "validation",
            "selection": "first_full_context_windows",
            "write_indices": routes,
            "prefix_lengths": prefix_lengths,
        },
    )
    reference_layer = model.stack.sdm_layers[
        next(iter(model.stack.sdm_layers))
    ]
    logical_banks = routes.shape[1] * reference_layer.memory_heads
    rows_per_bank = reference_layer.slots
    logical_rows_per_request = logical_banks * rows_per_bank
    rows = []
    for prefix in prefix_lengths:
        values, layer_values = route_occupancy_counts(
            routes,
            prefix=prefix,
            slots=rows_per_bank,
        )
        mean_bank_values = values / logical_banks
        rows.append(
            {
                "prefix_tokens": prefix,
                "unique_rows": describe(values),
                "mean_private_rows_per_bank": describe(mean_bank_values),
                "mean_active_fraction": float(values.mean() / logical_rows_per_request),
                "mean_private_value_bytes_fp32": float(
                    values.mean() * reference_layer.head_width * VALUE_ELEMENT_BYTES
                ),
                "mean_private_row_id_bytes": float(values.mean() * SLOT_ID_BYTES),
                "mean_rows_by_layer": layer_values.mean(0).tolist(),
            }
        )
    result = {
        "schema": "native-sdm-trained-occupancy-v2",
        "protocol_id": PROTOCOL_ID,
        "split": "validation",
        "selection": "first_full_context_windows",
        "examples": routes.shape[0],
        "logical_banks_per_request": logical_banks,
        "logical_rows_per_bank_n": rows_per_bank,
        "logical_rows_per_request": logical_rows_per_request,
        "prefixes": rows,
        "route_artifact": route_path.name,
        "route_artifact_sha256": sha256_file(route_path),
    }
    atomic_json(output / "TRAINED_OCCUPANCY.json", result)
    return result


def write_recovery(
    *,
    output: Path,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    manifest_sha256: str,
    metrics: list[dict[str, Any]],
    curve: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = output / "recovery_checkpoint.pt"
    atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "kind": RECOVERY_KIND,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "manifest_sha256": manifest_sha256,
            "metrics": metrics,
            "training_curve": curve,
            "health": json.loads((output / "HEALTHY.json").read_text(encoding="utf-8")),
            "execution_path": json.loads(
                (output / "EXECUTION_PATH.json").read_text(encoding="utf-8")
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        },
    )
    record = {
        "schema_version": 1,
        "status": "recoverable",
        "artifact": checkpoint.name,
        "step": step,
        "base_complete": step == config.steps,
        "sha256": sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }
    atomic_json(output / "RECOVERY.json", record)
    return record


def load_recovery(
    path: Path,
    *,
    expected_sha256: str,
    expected_config: dict[str, Any],
    expected_manifest_sha256: str,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("resume checkpoint SHA-256 changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != RECOVERY_KIND:
        raise ValueError("resume checkpoint kind changed")
    if payload.get("config") != expected_config:
        raise ValueError("resume checkpoint configuration changed")
    if payload.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("resume checkpoint data identity changed")
    step = int(payload.get("step", -1))
    if not 0 < step <= int(expected_config["steps"]):
        raise ValueError("resume checkpoint step is invalid")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if not isinstance(payload.get("health"), dict) or not isinstance(
        payload.get("execution_path"), dict
    ):
        raise ValueError("resume checkpoint lacks first-step execution evidence")
    return payload


def validate_config(args: argparse.Namespace, data: CanonicalWikiTextData) -> None:
    exact = {
        "protocol_id": PROTOCOL_ID,
        "passes": CANONICAL_PASSES,
        "steps": CANONICAL_STEPS,
        "schedule_steps": CANONICAL_STEPS,
        "batch_size": EFFECTIVE_BATCH_RECORDS,
        "micro_batch_size": 1,
        "warmup_steps": CANONICAL_WARMUP_STEPS,
        "seed": 0,
        "stream_seed": STREAM_SEED,
        "slots": 1_024,
        "reads": 16,
        "writes": 16,
        "memory_heads": 1,
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            raise ValueError(f"resolved {name}={getattr(args, name)!r}; expected {expected!r}")
    if data.training_steps != CANONICAL_STEPS or data.passes != CANONICAL_PASSES:
        raise ValueError("prepared data does not encode the canonical schedule")
    if args.reads != args.writes:
        raise ValueError("balanced sparse access requires W = R")
    if args.layout != "BBBBBBBA":
        raise ValueError("canonical Elastic layout changed")
    if args.activation_dtype != "bfloat16":
        raise ValueError("canonical activations must use bfloat16")
    if args.learning_rate != 3e-4 or args.weight_decay != 0.01:
        raise ValueError("canonical optimizer hyperparameters changed")
    if not math.isfinite(args.occupancy_price) or args.occupancy_price < 0:
        raise ValueError("occupancy price must be finite and nonnegative")
    for value in (
        args.expected_manifest_sha256,
        args.expected_data_payload_sha256,
        args.expected_double_build_sha256,
        args.expected_loader_sha256,
        args.campaign_design_sha256,
    ):
        if len(value) != 64:
            raise ValueError("campaign identities must be SHA-256 values")


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    try:
        import triton
    except ImportError as error:
        raise SystemExit("Triton is required") from error

    manifest_sha256 = sha256_file(args.manifest)
    if manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError("canonical WikiText manifest identity changed")
    double_build_path = args.manifest.with_name("DOUBLE_BUILD.json")
    if sha256_file(double_build_path) != args.expected_double_build_sha256:
        raise ValueError("canonical WikiText double-build identity changed")
    loader_path = Path(__file__).with_name("wikitext103_coverage.py")
    if sha256_file(loader_path) != args.expected_loader_sha256:
        raise ValueError("canonical WikiText loader identity changed")
    data = CanonicalWikiTextData(args.manifest, passes=args.passes)
    if data.manifest["remote_payload"]["sha256"] != args.expected_data_payload_sha256:
        raise ValueError("canonical WikiText payload identity changed")
    validate_config(args, data)

    config = TrainConfig(
        arm=args.arm,
        layout=args.layout,
        steps=args.steps,
        schedule_steps=args.schedule_steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        eval_batch_size=1,
        sequence_length=CONTEXT_LENGTH,
        vocab_size=VOCAB_SIZE,
        width=args.width,
        heads=args.heads,
        slots=args.slots,
        reads=args.reads,
        writes=args.writes,
        memory_heads=args.memory_heads,
        mlp_expansion=args.mlp_expansion,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        activation_dtype=args.activation_dtype,
        occupancy_price=args.occupancy_price,
    )
    set_seed(config.seed)
    device = torch.device("cuda")
    model = LanguageModel(
        vocab_size=config.vocab_size,
        maximum_sequence_length=config.sequence_length,
        layout=config.layout,
        width=config.width,
        heads=config.heads,
        slots=config.slots,
        reads=config.reads,
        writes=config.writes,
        memory_heads=config.memory_heads,
        mlp_expansion=config.mlp_expansion,
        activation_dtype=torch.bfloat16,
    )
    model.initialize_role_keyed(config.seed)
    model = model.to(device)
    accounting = parameter_accounting(model, config)
    if accounting["active_parameters"] != args.expected_active_parameters:
        raise ValueError("active parameter count differs from campaign declaration")

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            destination = no_decay if getattr(parameter, "_no_weight_decay", False) else decay
            destination.append(parameter)
    groups: list[dict[str, Any]] = [{"params": decay, "weight_decay": config.weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=True,
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if bool(args.resume_checkpoint) != bool(args.resume_checkpoint_sha256):
        raise ValueError("resume checkpoint and SHA-256 must be supplied together")
    resume_payload: dict[str, Any] | None = None
    if args.resume_checkpoint is not None:
        resume_payload = load_recovery(
            args.resume_checkpoint,
            expected_sha256=args.resume_checkpoint_sha256,
            expected_config=asdict(config),
            expected_manifest_sha256=manifest_sha256,
            model=model,
            optimizer=optimizer,
        )
    start_step = int(resume_payload["step"]) if resume_payload else 0
    metrics: list[dict[str, Any]] = list(resume_payload["metrics"]) if resume_payload else []
    curve: list[dict[str, Any]] = list(resume_payload["training_curve"]) if resume_payload else []

    coverage = data.coverage_report()
    configuration = {
        **asdict(config),
        "schema_version": 1,
        "campaign": args.campaign,
        "campaign_design_sha256": args.campaign_design_sha256,
        "model_identity": MODEL_IDENTITY,
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": manifest_sha256,
        "data_payload_sha256": data.manifest["remote_payload"]["sha256"],
        "double_build_sha256": args.expected_double_build_sha256,
        "loader_sha256": args.expected_loader_sha256,
        "checkpoint_steps": data.manifest["training"]["checkpoint_steps"],
        "parameter_accounting": accounting,
        "objective": {
            "task": "mean_next_token_nll_over_scored_targets",
            "pricing": "occupancy_price_times_straight_through_occupied_fraction",
            "normalization": "mean_over_requests_and_independent_sdm_banks",
            "coefficient_lambda": config.occupancy_price,
        },
        "resumed_from_step": start_step,
    }
    atomic_json(output / "config.json", configuration)
    atomic_json(output / "SDM_VALIDATION.json", verify_native_sdm(device, config))
    atomic_json(output / "ALLOCATOR_VALIDATION.json", verify_allocator(device, config))
    atomic_json(output / "CAUSALITY.json", verify_prefix_causality(model, config, device))
    recovered_checkpoint: dict[str, Any] | None = None
    if resume_payload is not None:
        atomic_json(output / "HEALTHY.json", resume_payload["health"])
        atomic_json(output / "EXECUTION_PATH.json", resume_payload["execution_path"])
        local_recovery = output / "recovery_checkpoint.pt"
        atomic_torch_save(local_recovery, resume_payload)
        recovered_checkpoint = {
            "schema_version": 1,
            "status": "recoverable",
            "artifact": local_recovery.name,
            "step": start_step,
            "base_complete": start_step == config.steps,
            "sha256": sha256_file(local_recovery),
            "bytes": local_recovery.stat().st_size,
            "restored_from_sha256": args.resume_checkpoint_sha256,
        }
        atomic_json(output / "RECOVERY.json", recovered_checkpoint)
        atomic_json(
            output / "RECOVERY_AUDIT.json",
            {
                "schema_version": 1,
                "interval_steps": args.recovery_checkpoint_interval,
                "checkpoints": [recovered_checkpoint],
                "latest": recovered_checkpoint,
            },
        )
    coverage.update(
        {
            "consumed_optimizer_steps": start_step,
            "consumed_target_presentations": data.target_presentations_through_step(start_step),
            "complete": start_step == config.steps,
        }
    )
    atomic_json(output / "COVERAGE.json", coverage)
    if resume_payload is not None:
        torch.set_rng_state(resume_payload["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume_payload["cuda_rng_states"])
    print(
        json.dumps(
            {
                "marker": "STARTED",
                "campaign": args.campaign,
                "arm": config.arm,
                "steps": config.steps,
                "target_presentations": data.coverage["total_target_presentations"],
                "reads": config.reads,
                "writes": config.writes,
                "occupancy_price": config.occupancy_price,
                "resumed_from_step": start_step,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    checkpoints = {int(step) for step in data.manifest["training"]["checkpoint_steps"]}
    recovery_steps = set(
        range(args.recovery_checkpoint_interval, config.steps + 1, args.recovery_checkpoint_interval)
    ) | {config.steps}
    recoveries: list[dict[str, Any]] = (
        [recovered_checkpoint] if recovered_checkpoint is not None else []
    )
    pending: list[dict[str, Any]] = []
    metric_path = output / "metrics.jsonl"
    curve_path = output / "training_curve.jsonl"
    started = time.perf_counter()
    interval_started = started
    interval_targets = 0
    optimizer_seconds = 0.0
    evaluation_seconds = 0.0
    recovery_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    with metric_path.open("w", encoding="utf-8") as metric_handle, curve_path.open(
        "w", encoding="utf-8"
    ) as curve_handle:
        for row in metrics:
            metric_handle.write(json.dumps(row, sort_keys=True) + "\n")
        for row in curve:
            curve_handle.write(json.dumps(row, sort_keys=True) + "\n")
        metric_handle.flush()
        curve_handle.flush()

        for step_index in range(start_step, config.steps):
            step = step_index + 1
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = data.train_batch(step_index)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            task_total = torch.zeros((), device=device)
            occupancy_total = torch.zeros((), device=device)
            objective_total = torch.zeros((), device=device)
            first_profile: torch.profiler.profile | None = None
            for row_index in range(config.batch_size):
                target_count = int(batch.target_mask[row_index].sum())
                values = to_device(
                    batch.tokens[row_index : row_index + 1, : target_count + 1],
                    device,
                )
                inputs, labels = values[:, :-1], values[:, 1:]
                profile = (
                    torch.profiler.profile(
                        activities=(
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        )
                    )
                    if step == 1 and row_index == 0
                    else None
                )
                profile_context = profile if profile is not None else nullcontext()
                with profile_context:
                    with torch.autocast("cuda", dtype=model.activation_dtype):
                        logits, routing = model(inputs, return_routing=True)
                        usage = aggregate_sdm_copy_on_write_accounting(
                            native_routing(routing), slots=config.slots
                        )
                        losses = F.cross_entropy(
                            logits.float().flatten(0, 1),
                            labels.flatten(),
                            reduction="sum",
                        )
                        task_contribution = losses / batch.target_count
                        occupancy_contribution = (
                            usage["straight_through_fraction"].mean() / config.batch_size
                        )
                        objective_contribution = (
                            task_contribution
                            + config.occupancy_price * occupancy_contribution
                        )
                    objective_contribution.backward()
                if profile is not None:
                    first_profile = profile
                task_total += task_contribution.detach()
                occupancy_total += occupancy_contribution.detach()
                objective_total += objective_contribution.detach()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            before = model.token_embedding.weight.detach().clone() if step == 1 else None
            optimizer.step()
            interval_targets += batch.target_count
            pending.append(
                {
                    "step": step,
                    "learning_rate": lr,
                    "train_nll": task_total,
                    "train_task_loss": task_total,
                    "train_hard_occupancy": occupancy_total,
                    "train_objective": objective_total,
                    "gradient_norm": gradient_norm.detach(),
                    "scored_targets": batch.target_count,
                    "pass_index": batch.pass_index,
                }
            )

            if step == 1:
                torch.cuda.synchronize(device)
                if before is None:
                    raise AssertionError("first-step parameter snapshot is absent")
                update = float((model.token_embedding.weight.detach() - before).abs().max())
                health = {
                    "schema_version": 1,
                    "status": "healthy",
                    "step": 1,
                    "task_nll": float(task_total),
                    "occupancy": float(occupancy_total),
                    "objective": float(objective_total),
                    "gradient_norm": float(gradient_norm),
                    "maximum_parameter_update": update,
                    "gpu": torch.cuda.get_device_name(device),
                    "reads": config.reads,
                    "writes": config.writes,
                }
                finite = all(
                    math.isfinite(float(health[key]))
                    for key in (
                        "task_nll",
                        "occupancy",
                        "objective",
                        "gradient_norm",
                        "maximum_parameter_update",
                    )
                )
                if not finite or update <= 0:
                    raise FloatingPointError(f"invalid first optimizer step: {health}")
                atomic_json(output / "HEALTHY.json", health)
                if first_profile is None:
                    raise AssertionError("first-step execution profile is absent")
                atomic_json(
                    output / "EXECUTION_PATH.json",
                    profiler_evidence(first_profile, config.layout.count("B")),
                )

            should_flush = (
                step % args.heartbeat_steps == 0
                or step in checkpoints
                or step in recovery_steps
            )
            if should_flush:
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                interval_elapsed = now - interval_started
                optimizer_seconds += interval_elapsed
                tensor_fields = (
                    "train_nll",
                    "train_task_loss",
                    "train_hard_occupancy",
                    "train_objective",
                    "gradient_norm",
                )
                pending_values = {
                    name: torch.stack([row[name].float() for row in pending]).cpu().tolist()
                    for name in tensor_fields
                }
                for index, pending_row in enumerate(pending):
                    curve_row = {
                        "step": pending_row["step"],
                        "learning_rate": pending_row["learning_rate"],
                        **{name: pending_values[name][index] for name in tensor_fields},
                        "scored_targets": pending_row["scored_targets"],
                        "target_presentations": data.target_presentations_through_step(
                            pending_row["step"]
                        ),
                        "pass_index": pending_row["pass_index"],
                    }
                    curve.append(curve_row)
                    curve_handle.write(json.dumps(curve_row, sort_keys=True) + "\n")
                curve_handle.flush()
                pending.clear()
                elapsed = now - started
                completed_targets = data.target_presentations_through_step(step)
                heartbeat = {
                    "schema_version": 1,
                    "marker": "HEARTBEAT",
                    "campaign": args.campaign,
                    "arm": config.arm,
                    "step": step,
                    "steps": config.steps,
                    "completed_passes": step / OPTIMIZER_STEPS_PER_PASS,
                    "target_presentations": completed_targets,
                    "total_target_presentations": data.coverage["total_target_presentations"],
                    "latest_train_nll": curve[-1]["train_nll"],
                    "latest_occupancy": curve[-1]["train_hard_occupancy"],
                    "latest_objective": curve[-1]["train_objective"],
                    "interval_targets_per_second": interval_targets / max(interval_elapsed, 1e-9),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / max(step - start_step, 1) * (config.steps - step),
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "reads": config.reads,
                    "writes": config.writes,
                }
                atomic_json(output / "HEARTBEAT.json", heartbeat)
                print(json.dumps(heartbeat, sort_keys=True), flush=True)
                interval_targets = 0
                interval_started = time.perf_counter()

            if step in checkpoints:
                evaluation_started = time.perf_counter()
                validation = evaluate(model, data, "validation", device)
                evaluation_seconds += time.perf_counter() - evaluation_started
                metric = {
                    "step": step,
                    "checkpoint_kind": (
                        "complete_pass"
                        if step % OPTIMIZER_STEPS_PER_PASS == 0
                        else "half_pass"
                    ),
                    "completed_passes": step / OPTIMIZER_STEPS_PER_PASS,
                    "target_presentations": data.target_presentations_through_step(step),
                    "train_nll": curve[-1]["train_nll"],
                    "train_hard_occupancy": curve[-1]["train_hard_occupancy"],
                    "train_objective": curve[-1]["train_objective"],
                    "gradient_norm": curve[-1]["gradient_norm"],
                    "learning_rate": lr,
                    "elapsed_seconds": time.perf_counter() - started,
                    "validation": validation,
                }
                metrics.append(metric)
                metric_handle.write(json.dumps(metric, sort_keys=True) + "\n")
                metric_handle.flush()
                print(json.dumps({"marker": "EVAL", **metric}, sort_keys=True), flush=True)
                interval_started = time.perf_counter()

            if step in recovery_steps:
                torch.cuda.synchronize(device)
                recovery_started = time.perf_counter()
                recovery = write_recovery(
                    output=output,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    manifest_sha256=manifest_sha256,
                    metrics=metrics,
                    curve=curve,
                )
                recovery_seconds += time.perf_counter() - recovery_started
                recoveries.append(recovery)
                atomic_json(
                    output / "RECOVERY_AUDIT.json",
                    {
                        "schema_version": 1,
                        "interval_steps": args.recovery_checkpoint_interval,
                        "checkpoints": recoveries,
                        "latest": recovery,
                    },
                )
                interval_started = time.perf_counter()

    complete_pass_metrics = {
        round(float(row["completed_passes"])): row
        for row in metrics
        if row["checkpoint_kind"] == "complete_pass"
    }
    if sorted(complete_pass_metrics) != [1, 2, 3]:
        raise ValueError("complete validation-by-pass series is absent")
    final_validation = complete_pass_metrics[CANONICAL_PASSES]["validation"]
    terminal_started = time.perf_counter()
    final_test = evaluate(model, data, "test", device)
    prefix_lengths = tuple(
        int(value) for value in args.serving_route_lengths.split(",") if value
    )
    trained_occupancy = export_trained_occupancy(
        model,
        data,
        device,
        examples=args.serving_route_examples,
        prefix_lengths=prefix_lengths,
        output=output,
    )
    evaluation_seconds += time.perf_counter() - terminal_started
    terminal_evaluation = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "checkpoint_rule": "terminal_after_predeclared_three_pass_coverage",
        "validation": final_validation,
        "test": final_test,
    }
    atomic_json(output / "TERMINAL_EVALUATION.json", terminal_evaluation)
    coverage.update(
        {
            "consumed_optimizer_steps": config.steps,
            "consumed_target_presentations": data.target_presentations_through_step(config.steps),
            "complete": True,
            "all_passes_exactly_once_without_replacement": True,
        }
    )
    atomic_json(output / "COVERAGE.json", coverage)
    total_seconds = time.perf_counter() - started
    segment_targets = (
        data.target_presentations_through_step(config.steps)
        - data.target_presentations_through_step(start_step)
    )
    result = {
        "schema": "elastic-sdm-canonical-wikitext103-v1",
        "status": "decision_bearing_complete",
        "campaign": args.campaign,
        "arm": config.arm,
        "seed": config.seed,
        "protocol_id": PROTOCOL_ID,
        "model_identity": MODEL_IDENTITY,
        "config": configuration,
        "manifest_sha256": manifest_sha256,
        "data_payload_sha256": data.manifest["remote_payload"]["sha256"],
        "parameter_accounting": accounting,
        "coverage": coverage,
        "validation_by_pass": {
            str(index): complete_pass_metrics[index]["validation"]
            for index in range(1, CANONICAL_PASSES + 1)
        },
        "half_pass_validation": next(
            row["validation"] for row in metrics if row["checkpoint_kind"] == "half_pass"
        ),
        "terminal_validation": final_validation,
        "terminal_test": final_test,
        "final_validation": final_validation,
        "final_test": final_test,
        "trained_occupancy": trained_occupancy,
        "optimizer_seconds": optimizer_seconds,
        "evaluation_seconds": evaluation_seconds,
        "recovery_checkpoint_seconds": recovery_seconds,
        "total_seconds": total_seconds,
        "optimizer_measurement_targets": segment_targets,
        "optimizer_targets_per_second": segment_targets / max(optimizer_seconds, 1e-9),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "curve_summary": {
            "complete_train_nll": curve_summary(curve, step_key="step", value_key="train_nll"),
            "complete_hard_occupancy": curve_summary(
                curve, step_key="step", value_key="train_hard_occupancy"
            ),
            "checkpoint_validation_nll": curve_summary(
                [
                    {"step": row["step"], "nll": row["validation"]["nll"]}
                    for row in metrics
                ],
                step_key="step",
                value_key="nll",
            ),
        },
        "metrics": metrics,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
    }
    atomic_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "marker": "COMPLETE",
                "campaign": args.campaign,
                "arm": config.arm,
                "validation_nll": final_validation["nll"],
                "test_nll": final_test["nll"],
                "mean_private_rows_per_bank_t2048": trained_occupancy["prefixes"][-1][
                    "mean_private_rows_per_bank"
                ]["mean"],
                "target_presentations": coverage["consumed_target_presentations"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-data-payload-sha256", required=True)
    parser.add_argument("--expected-double-build-sha256", required=True)
    parser.add_argument("--expected-loader-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-design-sha256", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--passes", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--schedule-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--heartbeat-steps", type=int, default=100)
    parser.add_argument("--serving-route-examples", type=int, default=64)
    parser.add_argument("--serving-route-lengths", default="16,64,256,1024,2048")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--slots", type=int, default=1_024)
    parser.add_argument("--reads", type=int, default=16)
    parser.add_argument("--writes", type=int, default=16)
    parser.add_argument("--memory-heads", type=int, default=1)
    parser.add_argument("--mlp-expansion", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=CANONICAL_WARMUP_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stream-seed", type=int, required=True)
    parser.add_argument("--activation-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--occupancy-price", type=float, default=0.0)
    parser.add_argument("--expected-active-parameters", type=int, required=True)
    parser.add_argument("--recovery-checkpoint-interval", type=int, default=2_400)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint-sha256")
    args = parser.parse_args()
    for name in (
        "passes",
        "steps",
        "schedule_steps",
        "batch_size",
        "micro_batch_size",
        "heartbeat_steps",
        "serving_route_examples",
        "recovery_checkpoint_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
