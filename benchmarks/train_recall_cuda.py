#!/usr/bin/env python3
"""Train one native-SDM hybrid on the complete adaptive recall suite."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.prepare_recall_suite import (
    MAXIMUM_SPAN,
    OUTPUT_CLASSES,
    OVERWRITE_MAP_OFFSET,
    OVERWRITE_MAP_TOKENS,
    OVERWRITE_QUERY_OFFSET,
    OVERWRITE_QUERY_TOKENS,
    POINTER_HOPS,
    POINTER_KEYS,
    POINTER_MAP_OFFSET,
    POINTER_MAP_TOKENS,
    POINTER_QUERY_OFFSET,
    POINTER_QUERY_TOKENS,
    QUERIES,
    SPAN_MAP_OFFSET,
    SPAN_MAP_TOKENS,
    SPAN_QUERY_OFFSET,
    SPAN_QUERY_TOKENS,
    SPAN_VALUES,
    VOCAB_SIZE,
    AdaptiveDepthData,
    sha256_file,
)
from benchmarks.train_wikitext_cuda import (
    atomic_json,
    atomic_torch_save,
    profiler_evidence,
    verify_allocator,
    verify_native_sdm,
    verify_prefix_causality,
)
from elastic_sdm.model import SDMDecoderStack
from elastic_sdm.sdm import aggregate_sdm_copy_on_write_accounting


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDENTITY = json.loads((ROOT / "PROJECT_IDENTITY.json").read_text())[
    "model_identity"
]
CHECKPOINT_CONDITIONS = (
    "pointer_h1_n192",
    "pointer_h2_n192",
    "pointer_h4_n192",
    "pointer_h8_n192",
    "span_s16_n32",
    "overwrite_v4_n32",
)
VALUE_ELEMENT_BYTES = 4
SLOT_ID_BYTES = 4


@dataclass(frozen=True)
class TrainConfig:
    arm: str
    layout: str
    steps: int
    schedule_steps: int
    batch_size: int
    micro_batch_size: int
    eval_batch_size: int
    maximum_sequence_length: int
    vocab_size: int
    output_classes: int
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


def reset_children(module: nn.Module, seed: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for child in module.modules():
            if child is module:
                continue
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()


class AdaptiveDepthEmbedding(nn.Module):
    """Expose shared identities while retaining source, value, and task roles."""

    _POINTER_MAPPING = 0
    _POINTER_QUERY = 1
    _SPAN_MAPPING = 2
    _SPAN_QUERY = 3
    _OVERWRITE_MAPPING = 4
    _OVERWRITE_QUERY = 5

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0 or width % 2:
            raise ValueError("recall embedding width must be positive and even")
        self.width = width
        self.half_width = width // 2
        self.identity = nn.Embedding(OUTPUT_CLASSES, self.half_width)
        self.slot = nn.Embedding(MAXIMUM_SPAN, self.half_width)
        self.role = nn.Embedding(6, width)
        self.hop = nn.Embedding(len(POINTER_HOPS), width)

    @staticmethod
    def _local(token_ids: torch.Tensor, offset: int, count: int) -> torch.Tensor:
        return (token_ids - offset).clamp(min=0, max=count - 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be [B,T]")
        first = torch.zeros(
            *token_ids.shape,
            self.half_width,
            device=token_ids.device,
            dtype=self.identity.weight.dtype,
        )
        second = torch.zeros_like(first)
        role_ids = torch.zeros_like(token_ids)
        hop_features = torch.zeros(
            *token_ids.shape,
            self.width,
            device=token_ids.device,
            dtype=self.identity.weight.dtype,
        )

        pointer_mapping = (token_ids >= POINTER_MAP_OFFSET) & (
            token_ids < POINTER_MAP_OFFSET + POINTER_MAP_TOKENS
        )
        local = self._local(token_ids, POINTER_MAP_OFFSET, POINTER_MAP_TOKENS)
        source = local // POINTER_KEYS
        target = local % POINTER_KEYS
        first = torch.where(
            pointer_mapping.unsqueeze(-1), self.identity(source), first
        )
        second = torch.where(
            pointer_mapping.unsqueeze(-1), self.identity(target), second
        )

        pointer_query = (token_ids >= POINTER_QUERY_OFFSET) & (
            token_ids < POINTER_QUERY_OFFSET + POINTER_QUERY_TOKENS
        )
        local = self._local(token_ids, POINTER_QUERY_OFFSET, POINTER_QUERY_TOKENS)
        hop_index = local // POINTER_KEYS
        start_key = local % POINTER_KEYS
        first = torch.where(
            pointer_query.unsqueeze(-1), self.identity(start_key), first
        )
        hop_features = torch.where(
            pointer_query.unsqueeze(-1), self.hop(hop_index), hop_features
        )
        role_ids = torch.where(
            pointer_query,
            torch.full_like(role_ids, self._POINTER_QUERY),
            role_ids,
        )

        span_mapping = (token_ids >= SPAN_MAP_OFFSET) & (
            token_ids < SPAN_MAP_OFFSET + SPAN_MAP_TOKENS
        )
        local = self._local(token_ids, SPAN_MAP_OFFSET, SPAN_MAP_TOKENS)
        value = local % SPAN_VALUES
        key_slot = local // SPAN_VALUES
        slot = key_slot % MAXIMUM_SPAN
        key = key_slot // MAXIMUM_SPAN
        first = torch.where(
            span_mapping.unsqueeze(-1),
            self.identity(key) + self.slot(slot),
            first,
        )
        second = torch.where(
            span_mapping.unsqueeze(-1), self.identity(value), second
        )
        role_ids = torch.where(
            span_mapping,
            torch.full_like(role_ids, self._SPAN_MAPPING),
            role_ids,
        )

        span_query = (token_ids >= SPAN_QUERY_OFFSET) & (
            token_ids < SPAN_QUERY_OFFSET + SPAN_QUERY_TOKENS
        )
        local = self._local(token_ids, SPAN_QUERY_OFFSET, SPAN_QUERY_TOKENS)
        slot = local % MAXIMUM_SPAN
        key = local // MAXIMUM_SPAN
        first = torch.where(
            span_query.unsqueeze(-1),
            self.identity(key) + self.slot(slot),
            first,
        )
        role_ids = torch.where(
            span_query,
            torch.full_like(role_ids, self._SPAN_QUERY),
            role_ids,
        )

        overwrite_mapping = (token_ids >= OVERWRITE_MAP_OFFSET) & (
            token_ids < OVERWRITE_MAP_OFFSET + OVERWRITE_MAP_TOKENS
        )
        local = self._local(
            token_ids, OVERWRITE_MAP_OFFSET, OVERWRITE_MAP_TOKENS
        )
        value = local % SPAN_VALUES
        key_slot = local // SPAN_VALUES
        slot = key_slot % MAXIMUM_SPAN
        key = key_slot // MAXIMUM_SPAN
        first = torch.where(
            overwrite_mapping.unsqueeze(-1),
            self.identity(key) + self.slot(slot),
            first,
        )
        second = torch.where(
            overwrite_mapping.unsqueeze(-1), self.identity(value), second
        )
        role_ids = torch.where(
            overwrite_mapping,
            torch.full_like(role_ids, self._OVERWRITE_MAPPING),
            role_ids,
        )

        overwrite_query = (token_ids >= OVERWRITE_QUERY_OFFSET) & (
            token_ids < OVERWRITE_QUERY_OFFSET + OVERWRITE_QUERY_TOKENS
        )
        local = self._local(
            token_ids, OVERWRITE_QUERY_OFFSET, OVERWRITE_QUERY_TOKENS
        )
        slot = local % MAXIMUM_SPAN
        key = local // MAXIMUM_SPAN
        first = torch.where(
            overwrite_query.unsqueeze(-1),
            self.identity(key) + self.slot(slot),
            first,
        )
        role_ids = torch.where(
            overwrite_query,
            torch.full_like(role_ids, self._OVERWRITE_QUERY),
            role_ids,
        )
        return torch.cat((first, second), dim=-1) + self.role(role_ids) + hop_features


class NativeRecallModel(nn.Module):
    """Structured recall wrapper around the native independent-layer stack."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.activation_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[config.activation_dtype]
        self.token_embedding = AdaptiveDepthEmbedding(config.width)
        self.position_embedding = nn.Embedding(
            config.maximum_sequence_length, config.width
        )
        self.stack = SDMDecoderStack(
            layout=config.layout,
            width=config.width,
            heads=config.heads,
            slots=config.slots,
            reads=config.reads,
            writes=config.writes,
            memory_heads=config.memory_heads,
            mlp_expansion=config.mlp_expansion,
        )
        self.final_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.output_classes, bias=False)

    def initialize_role_keyed(self, seed: int) -> None:
        base = seed * 100_000
        reset_children(self.token_embedding, base + 101)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base + 102)
            nn.init.normal_(
                self.position_embedding.weight,
                mean=0.0,
                std=self.stack.width**-0.5,
            )
            torch.manual_seed(base + 103)
            nn.init.normal_(
                self.output.weight,
                mean=0.0,
                std=self.stack.width**-0.5,
            )
        self.final_norm.reset_parameters()
        for physical_layer, kind in enumerate(self.stack.layout):
            if kind == "A":
                block = self.stack.attention_layers[str(physical_layer)]
                reset_children(block.mlp, base + 1_000 + physical_layer)
                reset_children(block.attention, base + 2_000 + physical_layer)
            else:
                self.stack.sdm_layers[str(physical_layer)].reset_role_keyed(
                    base + 6_000 + physical_layer
                )
                reset_children(
                    self.stack.sdm_mlps[str(physical_layer)],
                    base + 1_000 + physical_layer,
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_routing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[Any]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        time_steps = input_ids.shape[1]
        if time_steps > self.config.maximum_sequence_length:
            raise ValueError("sequence exceeds maximum_sequence_length")
        positions = torch.arange(time_steps, device=input_ids.device)
        tokens = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions).unsqueeze(0)
        ).to(self.activation_dtype)
        stacked = self.stack(tokens, return_routing=return_routing)
        if return_routing:
            hidden, routing = stacked
            return self.output(self.final_norm(hidden)), routing
        return self.output(self.final_norm(stacked))


def native_routing(rows: list[Any]) -> list[Any]:
    routed = [row.sdm for row in rows if row.sdm is not None]
    if len(routed) != sum(row.kind == "B" for row in rows):
        raise AssertionError("native SDM routing diagnostics are incomplete")
    return routed


def parameter_accounting(
    model: NativeRecallModel, config: TrainConfig
) -> dict[str, Any]:
    learned_ids = {
        id(parameter)
        for parameter in model.parameters()
        if getattr(parameter, "_sdm_memory_bank", False)
    }
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
    return {
        "active_parameters": active,
        "learned_initial_state_parameters": learned,
        "total_trainable_parameters": active + learned,
        "sdm_layers": layers,
        "logical_mutable_state_elements_per_sequence": elements,
        "logical_mutable_state_bytes_fp32_per_sequence": 4 * elements,
        "allocator_map_bytes_per_sequence": (
            4 * layers * config.memory_heads * config.slots
        ),
    }


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(
        np.asarray(array).astype(np.int64, copy=False)
    ).to(device, non_blocking=True)


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
    candidates = (
        250,
        500,
        1_000,
        2_000,
        4_000,
        6_000,
        8_000,
        10_000,
        12_500,
        15_000,
        17_500,
        20_000,
        22_500,
        25_000,
        27_500,
        total,
    )
    return sorted({step for step in candidates if 0 < step <= total})


def training_interval_stats(
    *,
    started: float,
    finished: float,
    examples: int,
    tokens: int,
) -> dict[str, float]:
    """Close one optimizer-active interval without checkpoint-time leakage."""

    if finished < started:
        raise ValueError("training interval finished before it started")
    if examples < 0 or tokens < 0:
        raise ValueError("training interval counts must be non-negative")
    elapsed = finished - started
    return {
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / max(elapsed, 1e-9),
        "tokens_per_second": tokens / max(elapsed, 1e-9),
    }


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


def describe(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def position_curve(values: torch.Tensor) -> dict[str, list[float]]:
    values = values.float()
    return {
        "mean": values.mean(dim=0).tolist(),
        "median": values.median(dim=0).values.tolist(),
        "p95": torch.quantile(values, 0.95, dim=0).tolist(),
        "maximum": values.max(dim=0).values.tolist(),
    }


def distance_quartiles(
    distances: np.ndarray, correct: np.ndarray
) -> list[dict[str, Any]]:
    order = np.argsort(distances, kind="stable")
    rows = []
    for index, selected in enumerate(np.array_split(order, 4)):
        selected_distances = distances[selected]
        rows.append(
            {
                "quartile": index + 1,
                "definition": "equal-query-count rank by dependency distance",
                "minimum_distance": int(selected_distances.min()),
                "maximum_distance": int(selected_distances.max()),
                "queries": int(len(selected)),
                "accuracy": float(correct[selected].mean()),
            }
        )
    return rows


def dependency_distances(
    tokens: np.ndarray, condition: dict[str, Any]
) -> np.ndarray:
    memory_tokens = int(condition["memory_tokens"])
    query_positions = (
        memory_tokens + np.arange(QUERIES, dtype=np.int64)
    )[None, :]
    family = str(condition["family"])
    if family == "pointer_chase":
        associations = int(condition["parameters"]["associations"])
        hops = int(condition["parameters"]["hops"])
        local = tokens[:, :associations] - POINTER_MAP_OFFSET
        sources = local // POINTER_KEYS
        targets = local % POINTER_KEYS
        query_local = tokens[:, associations:] - POINTER_QUERY_OFFSET
        current = query_local % POINTER_KEYS
        rows = np.arange(len(tokens))[:, None]
        positions = np.zeros((len(tokens), POINTER_KEYS), dtype=np.int64)
        table = np.zeros((len(tokens), POINTER_KEYS), dtype=np.uint32)
        positions[rows, sources] = np.arange(associations)[None, :]
        table[rows, sources] = targets
        maximum = np.zeros_like(current, dtype=np.int64)
        for _ in range(hops):
            dependency_position = positions[rows, current]
            maximum = np.maximum(maximum, query_positions - dependency_position)
            current = table[rows, current]
        return maximum

    mapping_offset = (
        SPAN_MAP_OFFSET if family == "span_recall" else OVERWRITE_MAP_OFFSET
    )
    query_offset = (
        SPAN_QUERY_OFFSET if family == "span_recall" else OVERWRITE_QUERY_OFFSET
    )
    parameters = condition["parameters"]
    associations = int(parameters["associations"])
    span_length = int(parameters["span_length"])
    versions = int(parameters.get("versions", 1))
    local = tokens[:, :memory_tokens] - mapping_offset
    key_slots = local // SPAN_VALUES
    keys = (key_slots // MAXIMUM_SPAN).reshape(
        len(tokens), versions, associations, span_length
    )[:, -1]
    slots = (key_slots % MAXIMUM_SPAN).reshape(
        len(tokens), versions, associations, span_length
    )[:, -1]
    start = memory_tokens - associations * span_length
    mapping_positions = np.arange(start, memory_tokens, dtype=np.int64)[
        None, :, None
    ]
    position_table = np.zeros(
        (len(tokens), OUTPUT_CLASSES, MAXIMUM_SPAN), dtype=np.int64
    )
    rows = np.arange(len(tokens))[:, None, None]
    position_table[rows, keys, slots] = mapping_positions.reshape(
        1, associations, span_length
    )
    query_local = tokens[:, memory_tokens:] - query_offset
    dependencies = position_table[
        np.arange(len(tokens))[:, None],
        query_local // MAXIMUM_SPAN,
        query_local % MAXIMUM_SPAN,
    ]
    return query_positions - dependencies


def state_accounting(
    model: NativeRecallModel, config: TrainConfig
) -> dict[str, Any]:
    layers = len(model.stack.sdm_layers)
    bank_width = config.width // config.memory_heads
    logical_rows = layers * config.memory_heads * config.slots
    learned = sum(
        parameter.numel()
        for parameter in model.parameters()
        if getattr(parameter, "_sdm_memory_bank", False)
    )
    return {
        "physical_sdm_layers": layers,
        "memory_heads_per_layer": config.memory_heads,
        "logical_slots_per_head": config.slots,
        "logical_rows_across_layers": logical_rows,
        "logical_dense_value_bytes_fp32": (
            logical_rows * bank_width * VALUE_ELEMENT_BYTES
        ),
        "shared_learned_initial_parameters": learned,
        "mutable_value_bytes_per_unique_row_fp32": (
            bank_width * VALUE_ELEMENT_BYTES
        ),
        "row_id_bytes_per_unique_row": SLOT_ID_BYTES,
        "mutable_overlay_bytes_per_unique_row": (
            bank_width * VALUE_ELEMENT_BYTES + SLOT_ID_BYTES
        ),
    }


@torch.no_grad()
def evaluate_condition(
    model: NativeRecallModel,
    data: AdaptiveDepthData,
    split: str,
    condition_id: str,
    device: torch.device,
    batch_size: int,
    *,
    maximum_examples: int | None,
    include_position_curve: bool,
) -> dict[str, Any]:
    model.eval()
    condition = data.condition(condition_id)
    tokens, labels = data.evaluation(split, condition_id)
    stop_at = len(tokens) if maximum_examples is None else min(
        len(tokens), maximum_examples
    )
    groups = int(condition["groups"])
    group_size = int(condition["group_size"])
    loss_sum = 0.0
    correct_count = 0
    exact_set_count = 0
    exact_group_count = 0
    correct_by_ordinal = np.zeros(QUERIES, dtype=np.int64)
    distance_rows: list[np.ndarray] = []
    correct_rows: list[np.ndarray] = []
    unique_rows: list[torch.Tensor] = []
    unique_layer_rows: list[torch.Tensor] = []
    slot_count_rows: list[torch.Tensor] = []
    first_touches = 0
    repeated_writes = 0
    private_reads = 0
    private_read_weight = 0.0
    route_entropy_sum = 0.0
    route_entropy_count = 0
    read_selections_per_position = 0
    read_weight_per_position = 0
    write_selections_per_position = 0
    layers = 0
    started = time.perf_counter()
    for start in range(0, stop_at, batch_size):
        stop = min(start + batch_size, stop_at)
        host_tokens = np.asarray(tokens[start:stop])
        batch_tokens = to_device(host_tokens, device)
        batch_labels = to_device(labels[start:stop], device)
        with torch.autocast("cuda", dtype=model.activation_dtype):
            full_logits, routing = model(batch_tokens, return_routing=True)
            usage = aggregate_sdm_copy_on_write_accounting(
                native_routing(routing), slots=model.config.slots
            )
            logits = full_logits[:, -QUERIES:].float()
        predictions = logits.argmax(dim=-1)
        correct = predictions == batch_labels
        loss_sum += float(
            F.cross_entropy(
                logits.reshape(-1, model.config.output_classes),
                batch_labels.reshape(-1),
                reduction="sum",
            )
        )
        correct_count += int(correct.sum())
        exact_set_count += int(correct.all(dim=-1).sum())
        exact_group_count += int(
            correct.reshape(-1, groups, group_size).all(dim=-1).sum()
        )
        correct_by_ordinal += correct.sum(dim=0).cpu().numpy().astype(np.int64)
        distance_rows.append(dependency_distances(host_tokens, condition).reshape(-1))
        correct_rows.append(correct.cpu().numpy().reshape(-1))
        unique_rows.append(usage["unique_by_position"].detach().cpu())
        unique_layer_rows.append(usage["unique_final_by_layer"].detach().cpu())
        slot_count_rows.append(usage["slot_write_counts"].detach().cpu())
        first_touches += int(usage["first_touch_by_position"].sum())
        repeated_writes += int(usage["repeated_write_by_position"].sum())
        private_reads += int(usage["private_read_count_by_position"].sum())
        private_read_weight += float(
            usage["private_read_weight_by_position"].sum()
        )
        entropy = usage["route_entropy_by_position"]
        route_entropy_sum += float(entropy.sum())
        route_entropy_count += entropy.numel()
        read_selections_per_position = int(usage["read_selections_per_position"])
        read_weight_per_position = int(usage["read_weight_per_position"])
        write_selections_per_position = int(
            usage["write_selections_per_position"]
        )
        layers = int(usage["layers"])
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    distance_values = np.concatenate(distance_rows)
    correct_values = np.concatenate(correct_rows)
    total_queries = stop_at * QUERIES
    total_groups = stop_at * groups
    result: dict[str, Any] = {
        "condition_id": condition_id,
        "family": condition["family"],
        "parameters": condition["parameters"],
        "sequence_length": condition["sequence_length"],
        "memory_tokens": condition["memory_tokens"],
        "query_loss": loss_sum / total_queries,
        "query_accuracy": correct_count / total_queries,
        "exact_group_accuracy": exact_group_count / total_groups,
        "exact_group_size": group_size,
        "exact_set_accuracy": exact_set_count / stop_at,
        "exact_set_size": QUERIES,
        "accuracy_by_query_ordinal": [
            {"query_ordinal": index + 1, "accuracy": int(value) / stop_at}
            for index, value in enumerate(correct_by_ordinal)
        ],
        "accuracy_by_dependency_distance_quartile": distance_quartiles(
            distance_values, correct_values
        ),
        "minimum_dependency_distance": int(distance_values.min()),
        "maximum_dependency_distance": int(distance_values.max()),
        "mean_dependency_distance": float(distance_values.mean()),
        "queries": total_queries,
        "groups": total_groups,
        "examples": stop_at,
        "seconds": elapsed,
        "queries_per_second": total_queries / elapsed,
    }
    unique = torch.cat(unique_rows)
    unique_final = unique[:, -1]
    memory_boundary = unique[:, int(condition["memory_tokens"]) - 1]
    unique_by_layer = torch.cat(unique_layer_rows)
    slot_counts = torch.cat(slot_count_rows).sum(dim=0).float()
    total_writes = first_touches + repeated_writes
    sequence_length = unique.shape[1]
    state = state_accounting(model, model.config)
    value_bytes = int(state["mutable_value_bytes_per_unique_row_fp32"])
    overlay_bytes = int(state["mutable_overlay_bytes_per_unique_row"])
    routing_layers: list[dict[str, Any]] = []
    entropy_denominator = math.log(model.config.slots)
    for layer in range(slot_counts.shape[0]):
        head_counts = slot_counts[layer]
        probabilities = head_counts / head_counts.sum(
            dim=-1, keepdim=True
        ).clamp_min(1)
        entropy = -(
            probabilities.clamp_min(1e-12)
            * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        mean_count = head_counts.mean(dim=-1)
        routing_layers.append(
            {
                "physical_sdm_index": layer,
                "normalized_slot_entropy": float(
                    (entropy / max(entropy_denominator, 1.0)).mean()
                ),
                "effective_slots": float(entropy.exp().mean()),
                "max_over_mean_slot_writes": float(
                    (
                        head_counts.max(dim=-1).values
                        / mean_count.clamp_min(1e-12)
                    ).mean()
                ),
                "unused_slot_fraction": float(head_counts.eq(0).float().mean()),
            }
        )
    capacity = layers * model.config.memory_heads * model.config.slots
    cow: dict[str, Any] = {
        "physical_sdm_layers": layers,
        "unique_mutable_rows_after_memory": describe(memory_boundary),
        "unique_mutable_rows_final": describe(unique_final),
        "unique_mutable_rows_final_by_layer_mean": (
            unique_by_layer.float().mean(dim=0).tolist()
        ),
        "active_fraction_after_memory": describe(memory_boundary / capacity),
        "active_fraction_final": describe(unique_final / capacity),
        "private_value_bytes_after_memory": describe(
            memory_boundary * value_bytes
        ),
        "private_value_bytes_final": describe(unique_final * value_bytes),
        "mutable_overlay_bytes_after_memory": describe(
            memory_boundary * overlay_bytes
        ),
        "mutable_overlay_bytes_final": describe(unique_final * overlay_bytes),
        "first_touch_rate": first_touches / max(total_writes, 1),
        "repeated_write_rate": repeated_writes / max(total_writes, 1),
        "private_read_selection_fraction": private_reads
        / max(stop_at * sequence_length * read_selections_per_position, 1),
        "private_read_weight_fraction": private_read_weight
        / max(stop_at * sequence_length * read_weight_per_position, 1),
        "mean_selected_write_route_entropy": route_entropy_sum
        / max(route_entropy_count, 1),
        "writes_per_position_across_layers": write_selections_per_position,
        "routing_by_layer": routing_layers,
    }
    if include_position_curve:
        cow["unique_mutable_rows_by_sequence_position"] = position_curve(unique)
    result["copy_on_write"] = cow
    return result


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    queries = sum(int(row["queries"]) for row in rows)
    groups = sum(int(row["groups"]) for row in rows)
    examples = sum(int(row["examples"]) for row in rows)
    return {
        "query_loss": sum(
            float(row["query_loss"]) * int(row["queries"]) for row in rows
        )
        / queries,
        "query_accuracy": sum(
            float(row["query_accuracy"]) * int(row["queries"]) for row in rows
        )
        / queries,
        "exact_group_accuracy": sum(
            float(row["exact_group_accuracy"]) * int(row["groups"]) for row in rows
        )
        / groups,
        "exact_set_accuracy": sum(
            float(row["exact_set_accuracy"]) * int(row["examples"]) for row in rows
        )
        / examples,
        "queries": queries,
        "groups": groups,
        "examples": examples,
    }


@torch.no_grad()
def evaluate_suite(
    model: NativeRecallModel,
    data: AdaptiveDepthData,
    split: str,
    device: torch.device,
    batch_size: int,
    *,
    maximum_examples: int | None,
    condition_ids: tuple[str, ...] | None = None,
    include_position_curve: bool = False,
) -> dict[str, Any]:
    selected = (
        tuple(str(row["id"]) for row in data.conditions)
        if condition_ids is None
        else condition_ids
    )
    rows = [
        evaluate_condition(
            model,
            data,
            split,
            condition_id,
            device,
            batch_size,
            maximum_examples=maximum_examples,
            include_position_curve=include_position_curve,
        )
        for condition_id in selected
    ]
    families = sorted({str(row["family"]) for row in rows})
    return {
        "split": split,
        "condition_ids": list(selected),
        "by_condition": rows,
        "by_family": {
            family: aggregate_rows(
                [row for row in rows if row["family"] == family]
            )
            for family in families
        },
        "aggregate": aggregate_rows(rows),
    }


def validate_condition_ids(data: AdaptiveDepthData) -> None:
    missing = set(CHECKPOINT_CONDITIONS) - set(data.condition_by_id)
    if missing:
        raise ValueError(f"checkpoint conditions are absent: {sorted(missing)}")


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    try:
        import triton
    except ImportError as error:
        raise SystemExit("Triton is required") from error
    if sha256_file(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("recall manifest SHA-256 mismatch")
    data = AdaptiveDepthData(args.manifest)
    data.verify_semantics()
    validate_condition_ids(data)
    manifest = data.manifest
    if args.steps > int(manifest["steps"]):
        raise ValueError("steps exceed deterministic stream length")
    if args.batch_size != int(manifest["batch_size"]):
        raise ValueError("batch size must match deterministic stream")
    if args.batch_size % args.micro_batch_size:
        raise ValueError("micro batch size must divide batch size")
    if not args.layout or any(kind not in "AB" for kind in args.layout):
        raise ValueError("layout must contain only A and B")
    if len(args.layout) != args.layers:
        raise ValueError("layout length must equal physical layers")
    if not args.layout.count("B"):
        raise ValueError("native SDM recall requires at least one B layer")
    if args.schedule_steps < args.steps:
        raise ValueError("schedule cannot be shorter than training")
    if args.occupancy_price < 0:
        raise ValueError("occupancy price must be nonnegative")
    config = TrainConfig(
        arm=args.arm,
        layout=args.layout,
        steps=args.steps,
        schedule_steps=args.schedule_steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        eval_batch_size=args.eval_batch_size,
        maximum_sequence_length=int(manifest["maximum_sequence_length"]),
        vocab_size=int(manifest["vocab_size"]),
        output_classes=int(manifest["output_classes"]),
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
    if config.vocab_size != VOCAB_SIZE or config.output_classes != OUTPUT_CLASSES:
        raise ValueError("recall vocabulary mismatch")
    set_seed(config.seed)
    device = torch.device("cuda")
    model = NativeRecallModel(config)
    model.initialize_role_keyed(config.seed)
    model = model.to(device)
    accounting = parameter_accounting(model, config)
    if (
        args.expected_active_parameters is not None
        and accounting["active_parameters"] != args.expected_active_parameters
    ):
        raise ValueError("active parameter count differs from campaign declaration")
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(
                parameter
            )
    groups: list[dict[str, Any]] = [
        {"params": decay, "weight_decay": config.weight_decay}
    ]
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
    atomic_json(
        output / "config.json",
        {
            **asdict(config),
            "model_identity": MODEL_IDENTITY,
            "manifest_sha256": sha256_file(args.manifest),
            "parameter_accounting": accounting,
            "state_accounting": state_accounting(model, config),
            "objective": (
                "query_loss + occupancy_price * "
                "ST(unique_first_written_rows / logical_rows)"
            ),
        },
    )
    atomic_json(output / "SDM_VALIDATION.json", verify_native_sdm(device, config))
    atomic_json(output / "ALLOCATOR_VALIDATION.json", verify_allocator(device, config))
    atomic_json(output / "CAUSALITY.json", verify_prefix_causality(model, config, device))

    metrics: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    pending: list[tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    recoveries: list[dict[str, Any]] = []
    checkpoints = checkpoint_steps(config.steps)
    recovery_steps = {
        *range(
            args.recovery_checkpoint_interval,
            config.steps + 1,
            args.recovery_checkpoint_interval,
        ),
        config.steps,
    }
    metric_path = output / "metrics.jsonl"
    curve_path = output / "training_curve.jsonl"
    started = time.perf_counter()
    interval_started = started
    interval_examples = 0
    interval_tokens = 0
    optimizer_seconds = 0.0
    evaluation_seconds = 0.0
    recovery_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    with metric_path.open("w") as metric_handle, curve_path.open("w") as curve_handle:
        for step_index in range(config.steps):
            step = step_index + 1
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch, labels, condition = data.train_batch(step_index)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            task_total = torch.zeros((), device=device)
            occupancy_total = torch.zeros((), device=device)
            objective_total = torch.zeros((), device=device)
            first_profile: torch.profiler.profile | None = None
            for start in range(0, config.batch_size, config.micro_batch_size):
                stop = start + config.micro_batch_size
                inputs = to_device(batch[start:stop], device)
                targets = to_device(labels[start:stop], device)
                profile = (
                    torch.profiler.profile(
                        activities=(
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        )
                    )
                    if step == 1 and start == 0
                    else None
                )
                profile_context = profile if profile is not None else nullcontext()
                with profile_context:
                    with torch.autocast("cuda", dtype=model.activation_dtype):
                        full_logits, routing = model(inputs, return_routing=True)
                        logits = full_logits[:, -QUERIES:].float()
                        usage = aggregate_sdm_copy_on_write_accounting(
                            native_routing(routing), slots=config.slots
                        )
                        task_loss = F.cross_entropy(
                            logits.reshape(-1, config.output_classes),
                            targets.reshape(-1),
                        )
                        occupancy = usage["straight_through_fraction"].mean()
                        objective = (
                            task_loss + config.occupancy_price * occupancy
                        )
                        scaled = objective * (
                            config.micro_batch_size / config.batch_size
                        )
                    scaled.backward()
                if profile is not None:
                    first_profile = profile
                scale = config.micro_batch_size / config.batch_size
                task_total += task_loss.detach() * scale
                occupancy_total += occupancy.detach() * scale
                objective_total += objective.detach() * scale
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            before = model.output.weight.detach().clone() if step == 1 else None
            optimizer.step()
            interval_examples += config.batch_size
            interval_tokens += config.batch_size * int(condition["sequence_length"])
            pending.append((step, lr, task_total, occupancy_total, objective_total))

            if step == 1:
                torch.cuda.synchronize(device)
                update = float((model.output.weight.detach() - before).abs().max())
                health = {
                    "schema_version": 1,
                    "status": "healthy",
                    "step": 1,
                    "task_loss": float(task_total),
                    "occupancy": float(occupancy_total),
                    "objective": float(objective_total),
                    "gradient_norm": float(gradient_norm),
                    "maximum_parameter_update": update,
                    "gpu": torch.cuda.get_device_name(device),
                }
                if not all(
                    math.isfinite(float(health[key]))
                    for key in (
                        "task_loss",
                        "occupancy",
                        "objective",
                        "gradient_norm",
                        "maximum_parameter_update",
                    )
                ) or update <= 0:
                    raise FloatingPointError(f"invalid first optimizer step: {health}")
                atomic_json(output / "HEALTHY.json", health)
                if first_profile is None:
                    raise AssertionError("first-step execution profile is absent")
                atomic_json(
                    output / "EXECUTION_PATH.json",
                    profiler_evidence(first_profile, config.layout.count("B")),
                )

            if step in checkpoints:
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                interval = training_interval_stats(
                    started=interval_started,
                    finished=now,
                    examples=interval_examples,
                    tokens=interval_tokens,
                )
                interval_elapsed = interval["elapsed_seconds"]
                optimizer_seconds += interval_elapsed
                stacked = torch.stack(
                    [
                        torch.stack((row[2].float(), row[3].float(), row[4].float()))
                        for row in pending
                    ]
                ).cpu()
                for pending_row, values in zip(pending, stacked, strict=True):
                    curve_row = {
                        "step": pending_row[0],
                        "learning_rate": pending_row[1],
                        "train_task_loss": float(values[0]),
                        "train_hard_occupancy": float(values[1]),
                        "train_objective": float(values[2]),
                    }
                    curve.append(curve_row)
                    curve_handle.write(json.dumps(curve_row, sort_keys=True) + "\n")
                curve_handle.flush()
                pending.clear()
                evaluation_started = time.perf_counter()
                validation = evaluate_suite(
                    model,
                    data,
                    "validation",
                    device,
                    config.eval_batch_size,
                    maximum_examples=args.checkpoint_eval_examples,
                    condition_ids=CHECKPOINT_CONDITIONS,
                )
                evaluation_seconds += time.perf_counter() - evaluation_started
                metric = {
                    "step": step,
                    "train_condition_id": condition["id"],
                    "train_family": condition["family"],
                    "train_task_loss": float(task_total),
                    "train_hard_occupancy": float(occupancy_total),
                    "train_objective": float(objective_total),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": lr,
                    "training_examples_per_second": interval[
                        "examples_per_second"
                    ],
                    "training_tokens_per_second": interval["tokens_per_second"],
                    "elapsed_seconds": now - started,
                    "validation": validation,
                }
                metrics.append(metric)
                metric_handle.write(json.dumps(metric, sort_keys=True) + "\n")
                metric_handle.flush()
                print(json.dumps(metric, sort_keys=True), flush=True)
                interval_started = time.perf_counter()
                interval_examples = 0
                interval_tokens = 0

            if step in recovery_steps:
                torch.cuda.synchronize(device)
                recovery_started = time.perf_counter()
                if interval_examples:
                    interval = training_interval_stats(
                        started=interval_started,
                        finished=recovery_started,
                        examples=interval_examples,
                        tokens=interval_tokens,
                    )
                    optimizer_seconds += interval["elapsed_seconds"]
                    interval_examples = 0
                    interval_tokens = 0
                checkpoint = output / "recovery_checkpoint.pt"
                atomic_torch_save(
                    checkpoint,
                    {
                        "schema_version": 1,
                        "kind": "elastic_sdm_recall_recovery",
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": asdict(config),
                        "step": step,
                        "manifest_sha256": sha256_file(args.manifest),
                        "metrics": metrics,
                        "training_curve": curve,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all(),
                    },
                )
                recovery_elapsed = time.perf_counter() - recovery_started
                recovery_seconds += recovery_elapsed
                recovery = {
                    "schema_version": 1,
                    "status": "recoverable",
                    "artifact": checkpoint.name,
                    "step": step,
                    "base_complete": step == config.steps,
                    "sha256": sha256_file(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "write_and_hash_seconds": recovery_elapsed,
                }
                recoveries.append(recovery)
                atomic_json(output / "RECOVERY.json", recovery)
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

    evaluation_started = time.perf_counter()
    final_validation = evaluate_suite(
        model,
        data,
        "validation",
        device,
        config.eval_batch_size,
        maximum_examples=args.final_eval_examples,
    )
    final_test = evaluate_suite(
        model,
        data,
        "test",
        device,
        config.eval_batch_size,
        maximum_examples=args.final_eval_examples,
        include_position_curve=True,
    )
    evaluation_seconds += time.perf_counter() - evaluation_started
    if args.save_checkpoint:
        atomic_torch_save(
            output / "final_checkpoint.pt",
            {
                "model": model.state_dict(),
                "config": asdict(config),
                "step": config.steps,
                "manifest_sha256": sha256_file(args.manifest),
            },
        )
    total_seconds = time.perf_counter() - started
    result = {
        "schema": "elastic-sdm-adaptive-recall-v1",
        "model_identity": MODEL_IDENTITY,
        "config": asdict(config),
        "manifest_sha256": sha256_file(args.manifest),
        "parameter_accounting": accounting,
        "state_accounting": state_accounting(model, config),
        "optimizer_seconds": optimizer_seconds,
        "evaluation_seconds": evaluation_seconds,
        "recovery_checkpoint_seconds": recovery_seconds,
        "total_seconds": total_seconds,
        "training_examples": config.steps * config.batch_size,
        "optimizer_examples_per_second": (
            config.steps * config.batch_size / optimizer_seconds
        ),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_validation": final_validation,
        "final_test": final_test,
        "curve_summary": {
            "complete_task_loss": curve_summary(
                curve, step_key="step", value_key="train_task_loss"
            ),
            "complete_hard_occupancy": curve_summary(
                curve, step_key="step", value_key="train_hard_occupancy"
            ),
            "checkpoint_query_loss": curve_summary(
                [
                    {
                        "step": row["step"],
                        "query_loss": row["validation"]["aggregate"]["query_loss"],
                    }
                    for row in metrics
                ],
                step_key="step",
                value_key="query_loss",
            ),
        },
        "training_curve": curve,
        "metrics": metrics,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
    }
    atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--schedule-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-eval-examples", type=int, default=256)
    parser.add_argument("--final-eval-examples", type=int, default=2_048)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--slots", type=int, default=256)
    parser.add_argument("--reads", type=int, default=16)
    parser.add_argument("--writes", type=int, default=16)
    parser.add_argument("--memory-heads", type=int, default=1)
    parser.add_argument("--mlp-expansion", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--activation-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--occupancy-price", type=float, default=0.0)
    parser.add_argument("--expected-active-parameters", type=int)
    parser.add_argument("--recovery-checkpoint-interval", type=int, default=5_000)
    parser.add_argument(
        "--save-checkpoint", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    for name in (
        "layers",
        "steps",
        "schedule_steps",
        "batch_size",
        "micro_batch_size",
        "eval_batch_size",
        "checkpoint_eval_examples",
        "final_eval_examples",
        "width",
        "heads",
        "slots",
        "reads",
        "writes",
        "memory_heads",
        "recovery_checkpoint_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
