#!/usr/bin/env python3
"""Shared validation and accounting for the canonical experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from elastic_sdm.allocator import (
    PackedSDMCopyOnWriteState,
    dense_sdm_sparse_step,
    packed_sdm_cow_step,
)
from elastic_sdm.model import LanguageModel
from elastic_sdm.sdm import (
    SparseDeltaMemory,
    dense_sparse_routes,
    gated_delta_recurrence,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDENTITY = json.loads((ROOT / "PROJECT_IDENTITY.json").read_text())[
    "model_identity"
]
VALUE_ELEMENT_BYTES = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class TrainConfig:
    arm: str
    layout: str
    steps: int
    schedule_steps: int
    batch_size: int
    micro_batch_size: int
    eval_batch_size: int
    sequence_length: int
    vocab_size: int
    width: int
    heads: int
    slots: int
    reads: int
    writes: int
    memory_heads: int
    mlp_expansion: float
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    seed: int
    activation_dtype: str
    occupancy_price: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.int64, copy=False)).to(
        device, non_blocking=True
    )


def learning_rate(step: int, config: TrainConfig) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(
        1, config.schedule_steps - config.warmup_steps
    )
    progress = min(max(progress, 0.0), 1.0)
    return config.learning_rate * (
        0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))
    )


def checkpoint_steps(total: int) -> list[int]:
    candidates = (250, 500, 1_000, 2_000, 3_000, 4_000, total)
    return sorted({step for step in candidates if 0 < step <= total})


def curve_summary(
    rows: list[dict[str, Any]], *, step_key: str, value_key: str
) -> dict[str, Any]:
    steps = [int(row[step_key]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    if not values:
        raise ValueError("curve cannot be empty")
    area = sum(
        0.5
        * (values[index - 1] + values[index])
        * (steps[index] - steps[index - 1])
        for index in range(1, len(values))
    )
    return {
        "points": len(values),
        "first_step": steps[0],
        "last_step": steps[-1],
        "mean": sum(values) / len(values),
        "normalized_auc": (
            values[0] if len(values) == 1 else area / (steps[-1] - steps[0])
        ),
    }


def parameter_accounting(model: LanguageModel, config: TrainConfig) -> dict[str, Any]:
    learned_ids = {
        id(parameter)
        for parameter in model.parameters()
        if getattr(parameter, "_sdm_memory_bank", False)
    }
    learned_names = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in learned_ids
    ]
    active = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in learned_ids
    )
    learned = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in learned_ids
    )
    layers = config.layout.count("B")
    elements = layers * config.slots * config.width
    map_elements = layers * config.memory_heads * config.slots
    return {
        "active_parameters": active,
        "learned_initial_state_parameters": learned,
        "learned_initial_state_names": learned_names,
        "total_trainable_parameters": active + learned,
        "logical_mutable_state_elements_per_sequence": elements,
        "logical_mutable_state_bytes_fp32_per_sequence": 4 * elements,
        "logical_mutable_state_bytes_bf16_per_sequence": 2 * elements,
        "allocator_map_bytes_per_sequence": 4 * map_elements,
        "sdm_layers": layers,
    }


def native_routing(rows: list[Any]) -> list[Any]:
    routed = [row.sdm for row in rows if row.sdm is not None]
    if len(routed) != sum(row.kind == "B" for row in rows):
        raise AssertionError("native SDM routing diagnostics are incomplete")
    return routed


def describe(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def config_read_weight_per_position(model: LanguageModel) -> int:
    return sum(
        module.memory_heads for module in model.stack.sdm_layers.values()
    )


def profiler_evidence(profile: torch.profiler.profile, layers: int) -> dict[str, Any]:
    events = {row.key: int(row.count) for row in profile.key_averages()}
    route_calls = int(events.get("sdm_position_parallel_route_generation", 0))
    occupancy_calls = int(events.get("sdm_copy_on_write_occupancy", 0))
    suspicious = sorted(
        key
        for key in events
        if "tokenwise" in key.lower() or "serial_copy_on_write" in key.lower()
    )
    if route_calls != layers or occupancy_calls != layers or suspicious:
        raise AssertionError(
            "native SDM execution path changed: "
            f"routes={route_calls}, occupancy={occupancy_calls}, suspicious={suspicious}"
        )
    return {
        "schema_version": 1,
        "evidence": "torch profiler over the first real optimizer microbatch",
        "position_parallel_route_generation_calls": route_calls,
        "vectorized_occupancy_accounting_calls": occupancy_calls,
        "expected_sdm_layers": layers,
        "tokenwise_controller_loop": False,
        "second_model_pass": False,
        "suspicious_events": suspicious,
    }


def verify_prefix_causality(
    model: LanguageModel, config: TrainConfig, device: torch.device
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(config.seed + 91_003)
    tokens = torch.randint(
        config.vocab_size, (2, 16), generator=generator, device=device
    )
    cut = 7
    changed = tokens.clone()
    changed[:, cut + 1 :] = torch.randint(
        config.vocab_size,
        changed[:, cut + 1 :].shape,
        generator=generator,
        device=device,
    )
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=model.activation_dtype):
        baseline = model(tokens)
        counterfactual = model(changed)
    difference = float(
        (baseline[:, : cut + 1].float() - counterfactual[:, : cut + 1].float())
        .abs()
        .max()
    )
    if difference != 0.0:
        raise AssertionError(f"prefix causality failed: {difference}")
    return {
        "schema_version": 1,
        "status": "causal",
        "maximum_prefix_difference": difference,
        "cut_position": cut,
    }


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    if float(delta.abs().max()) <= 1e-7:
        return 0.0
    return float(delta.norm() / expected.float().norm().clamp_min(1e-8))


def verify_native_sdm(device: torch.device, config: TrainConfig) -> dict[str, Any]:
    import copy

    width = min(config.width, 64)
    heads = config.memory_heads
    if width % heads:
        width = config.width
    fast = SparseDeltaMemory(
        width,
        slots=config.slots,
        reads=config.reads,
        writes=config.writes,
        memory_heads=heads,
    ).to(device)
    fast.reset_role_keyed(config.seed * 100_000 + 12_101)
    reference = copy.deepcopy(fast)
    source = torch.randn(1, 6, width, device=device, requires_grad=True)
    reference_source = source.detach().clone().requires_grad_(True)
    probe = torch.randn_like(source)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, routing = fast(
            source, return_routing=True, include_final_memory=True
        )
        serial_output, serial_routing = reference(
            reference_source,
            return_routing=True,
            include_final_memory=True,
            serial_reference=True,
        )
    # Match the production training loop: forward uses autocast, while
    # backward runs after leaving the autocast context so explicitly-fp32
    # recurrent intermediates remain fp32.
    (output.float() * probe).sum().backward()
    (serial_output.float() * probe).sum().backward()
    parameter_gradient = 0.0
    for (name, parameter), (reference_name, reference_parameter) in zip(
        fast.named_parameters(), reference.named_parameters(), strict=True
    ):
        if name != reference_name:
            raise AssertionError("native SDM parameter order changed")
        if parameter.grad is not None:
            parameter_gradient = max(
                parameter_gradient,
                relative_l2(parameter.grad, reference_parameter.grad),
            )
    if routing.final_memory is None or serial_routing.final_memory is None:
        raise AssertionError("SDM validation final state is absent")

    batch, time, memory_heads, _ = routing.write_indices.shape
    write_indices = routing.write_indices.permute(0, 2, 1, 3).flatten(0, 1)
    write_weights = routing.write_weights.permute(0, 2, 1, 3).flatten(0, 1)
    read_indices = routing.read_indices.permute(0, 2, 1, 3).flatten(0, 1)
    read_weights = routing.read_weights.permute(0, 2, 1, 3).flatten(0, 1)
    write_routes = dense_sparse_routes(write_weights, write_indices, config.slots)
    read_routes = dense_sparse_routes(read_weights, read_indices, config.slots)
    values = routing.values.permute(0, 2, 1, 3).flatten(0, 1)
    input_gate = routing.input_gate.permute(0, 2, 1).flatten(0, 1)
    forget = routing.forget_log_gate.permute(0, 2, 1).flatten(0, 1)
    initial = (
        fast.initial_memory.unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .flatten(0, 1)
        .to(values.dtype)
    )
    split = time // 2
    first, carried = gated_delta_recurrence(
        initial,
        write_routes[:, :split],
        values[:, :split],
        input_gate[:, :split],
        forget[:, :split],
        read_routes[:, :split],
    )
    second, split_final = gated_delta_recurrence(
        carried,
        write_routes[:, split:],
        values[:, split:],
        input_gate[:, split:],
        forget[:, split:],
        read_routes[:, split:],
    )
    full, full_final = gated_delta_recurrence(
        initial, write_routes, values, input_gate, forget, read_routes
    )
    checks = {
        "output_relative_l2": relative_l2(output, serial_output),
        "input_gradient_relative_l2": relative_l2(
            source.grad, reference_source.grad
        ),
        "parameter_gradient_max_relative_l2": parameter_gradient,
        "final_state_relative_l2": relative_l2(
            routing.final_memory, serial_routing.final_memory
        ),
        "chunk_output_relative_l2": relative_l2(
            torch.cat((first, second), dim=1), full
        ),
        "chunk_state_relative_l2": relative_l2(split_final, full_final),
    }
    if max(checks.values()) > 0.035:
        raise AssertionError(f"native SDM differs from serial reference: {checks}")
    return {
        "schema_version": 1,
        "status": "equivalent",
        "relative_l2_tolerance": 0.035,
        **checks,
        "model_identity": MODEL_IDENTITY,
    }


def verify_allocator(device: torch.device, config: TrainConfig) -> dict[str, Any]:
    banks = 3
    width = config.width // config.memory_heads
    initial = torch.randn(1, config.slots, width, device=device)
    dense = initial.expand(banks, -1, -1).clone().float()
    packed = PackedSDMCopyOnWriteState.allocate(
        initial,
        banks=banks,
        capacity_rows=banks * config.slots,
        state_dtype=torch.float32,
    )
    maximum_read_difference = 0.0
    maximum_state_difference = 0.0
    for step in range(3):
        write_indices = torch.stack(
            [
                (torch.arange(config.writes, device=device) + row + step)
                % config.slots
                for row in range(banks)
            ]
        )
        write_weights = torch.softmax(
            torch.randn(banks, config.writes, device=device), dim=-1
        )
        read_indices = torch.stack(
            [
                (torch.arange(config.reads, device=device) + row) % config.slots
                for row in range(banks)
            ]
        )
        read_weights = torch.softmax(
            torch.randn(banks, config.reads, device=device), dim=-1
        )
        values = torch.randn(banks, width, device=device)
        input_gate = torch.sigmoid(torch.randn(banks, device=device))
        forget = -torch.rand(banks, device=device)
        dense_read = dense_sdm_sparse_step(
            dense,
            write_indices,
            write_weights,
            values,
            input_gate,
            forget,
            read_indices,
            read_weights,
        )
        packed_read, _ = packed_sdm_cow_step(
            packed,
            write_indices,
            write_weights,
            values,
            input_gate,
            forget,
            read_indices,
            read_weights,
        )
        maximum_read_difference = max(
            maximum_read_difference,
            float((dense_read.float() - packed_read.float()).abs().max()),
        )
        maximum_state_difference = max(
            maximum_state_difference,
            float((dense - packed.materialize_dense()).abs().max()),
        )
    if maximum_read_difference != 0.0 or maximum_state_difference != 0.0:
        raise AssertionError("packed allocator changed native SDM state")
    return {
        "schema_version": 1,
        "status": "exact",
        "dense_packed_read_maximum_difference": maximum_read_difference,
        "dense_packed_state_maximum_difference": maximum_state_difference,
        "allocated_rows": int(packed.allocated_rows_tensor()),
        "invariants": packed.validate_invariants(),
    }


@torch.no_grad()
def route_occupancy_counts(
    routes: torch.Tensor,
    *,
    prefix: int,
    slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count distinct logical rows per example and layer without serial sorts."""

    if routes.ndim != 5 or not 0 < prefix <= routes.shape[2] or slots <= 0:
        raise ValueError("routes must be [E,L,T,H,W] with a valid prefix and slots")
    examples, layers, _, memory_heads, _ = routes.shape
    selected = routes[:, :, :prefix].permute(0, 1, 3, 2, 4)
    selected = selected.reshape(examples, layers, memory_heads, -1).long()
    active = torch.zeros(
        examples,
        layers,
        memory_heads,
        slots,
        dtype=torch.bool,
        device=routes.device,
    )
    active.scatter_(3, selected, True)
    by_layer = active.sum(dim=(2, 3)).to(torch.float32)
    return by_layer.sum(dim=1), by_layer
