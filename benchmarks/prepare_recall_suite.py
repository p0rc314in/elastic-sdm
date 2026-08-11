#!/usr/bin/env python3
"""Prepare an immutable multitask exact-recall stream for native SDM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


FORMAT = "elastic_sdm_adaptive_recall_suite_v1"

OUTPUT_CLASSES = 192
POINTER_KEYS = 192
POINTER_HOPS = (1, 2, 4, 8)
SPAN_KEYS = 64
SPAN_VALUES = 64
MAXIMUM_SPAN = 16
QUERIES = 16

POINTER_MAP_OFFSET = 0
POINTER_MAP_TOKENS = POINTER_KEYS * POINTER_KEYS
POINTER_QUERY_OFFSET = POINTER_MAP_OFFSET + POINTER_MAP_TOKENS
POINTER_QUERY_TOKENS = len(POINTER_HOPS) * POINTER_KEYS
SPAN_MAP_OFFSET = POINTER_QUERY_OFFSET + POINTER_QUERY_TOKENS
SPAN_MAP_TOKENS = SPAN_KEYS * MAXIMUM_SPAN * SPAN_VALUES
SPAN_QUERY_OFFSET = SPAN_MAP_OFFSET + SPAN_MAP_TOKENS
SPAN_QUERY_TOKENS = SPAN_KEYS * MAXIMUM_SPAN
OVERWRITE_MAP_OFFSET = SPAN_QUERY_OFFSET + SPAN_QUERY_TOKENS
OVERWRITE_MAP_TOKENS = SPAN_MAP_TOKENS
OVERWRITE_QUERY_OFFSET = OVERWRITE_MAP_OFFSET + OVERWRITE_MAP_TOKENS
OVERWRITE_QUERY_TOKENS = SPAN_QUERY_TOKENS
VOCAB_SIZE = OVERWRITE_QUERY_OFFSET + OVERWRITE_QUERY_TOKENS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(
    root: Path,
    path: Path,
    dtype: str,
    shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": dtype,
        "shape": list(shape),
    }


def build_conditions() -> tuple[dict[str, Any], ...]:
    conditions: list[dict[str, Any]] = []
    for associations in (32, 64, 128, 192):
        for hops in POINTER_HOPS:
            conditions.append(
                {
                    "id": f"pointer_h{hops}_n{associations}",
                    "family": "pointer_chase",
                    "parameters": {
                        "associations": associations,
                        "hops": hops,
                    },
                    "memory_tokens": associations,
                    "sequence_length": associations + QUERIES,
                    "queries": QUERIES,
                    "groups": QUERIES,
                    "group_size": 1,
                    "output_support": POINTER_KEYS,
                }
            )
    for associations in (16, 32):
        for span_length in (1, 4, 8, 16):
            query_keys = QUERIES // span_length
            conditions.append(
                {
                    "id": f"span_s{span_length}_n{associations}",
                    "family": "span_recall",
                    "parameters": {
                        "associations": associations,
                        "span_length": span_length,
                        "query_keys": query_keys,
                    },
                    "memory_tokens": associations * span_length,
                    "sequence_length": associations * span_length + QUERIES,
                    "queries": QUERIES,
                    "groups": query_keys,
                    "group_size": span_length,
                    "output_support": SPAN_VALUES,
                }
            )
    for associations in (16, 32):
        for versions in (1, 2, 4):
            span_length = 4
            query_keys = QUERIES // span_length
            conditions.append(
                {
                    "id": f"overwrite_v{versions}_n{associations}",
                    "family": "overwrite_recall",
                    "parameters": {
                        "associations": associations,
                        "versions": versions,
                        "span_length": span_length,
                        "query_keys": query_keys,
                    },
                    "memory_tokens": associations * span_length * versions,
                    "sequence_length": (
                        associations * span_length * versions + QUERIES
                    ),
                    "queries": QUERIES,
                    "groups": query_keys,
                    "group_size": span_length,
                    "output_support": SPAN_VALUES,
                }
            )
    for index, condition in enumerate(conditions):
        condition["index"] = index
    return tuple(conditions)


CONDITIONS = build_conditions()
CONDITION_BY_ID = {str(row["id"]): row for row in CONDITIONS}


def balanced_schedule(
    generator: np.random.Generator,
    condition_count: int,
    steps: int,
) -> np.ndarray:
    schedule = np.resize(
        np.arange(condition_count, dtype=np.uint16),
        steps,
    )
    generator.shuffle(schedule)
    return schedule


def _unique_keys(
    generator: np.random.Generator,
    batch_size: int,
    vocabulary: int,
    count: int,
) -> np.ndarray:
    scores = generator.random((batch_size, vocabulary))
    return np.argsort(scores, axis=1)[:, :count].astype(np.uint16)


def _pointer_batch(
    generator: np.random.Generator,
    batch_size: int,
    condition: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    associations = int(condition["parameters"]["associations"])
    hops = int(condition["parameters"]["hops"])
    hop_index = POINTER_HOPS.index(hops)
    keys = _unique_keys(
        generator,
        batch_size,
        POINTER_KEYS,
        associations,
    )
    rows = np.arange(batch_size)[:, None]
    cycle_order = np.argsort(
        generator.random((batch_size, associations)),
        axis=1,
    )
    cycle_keys = keys[rows, cycle_order]
    cycle_targets = np.roll(cycle_keys, -1, axis=1)
    table = np.zeros((batch_size, POINTER_KEYS), dtype=np.uint16)
    table[rows, cycle_keys] = cycle_targets
    mapping_order = np.argsort(
        generator.random((batch_size, associations)),
        axis=1,
    )
    sources = keys[rows, mapping_order]
    targets = table[rows, sources]
    query_order = np.argsort(
        generator.random((batch_size, associations)),
        axis=1,
    )[:, :QUERIES]
    query_keys = keys[rows, query_order]

    labels = query_keys.copy()
    for _ in range(hops):
        labels = table[rows, labels]

    mappings = (
        POINTER_MAP_OFFSET + sources.astype(np.uint32) * POINTER_KEYS + targets
    )
    queries = (
        POINTER_QUERY_OFFSET
        + hop_index * POINTER_KEYS
        + query_keys.astype(np.uint32)
    )
    tokens = np.concatenate((mappings, queries), axis=1)
    return tokens.astype(np.uint32, copy=False), labels.astype(np.uint8)


def _span_batch(
    generator: np.random.Generator,
    batch_size: int,
    condition: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    parameters = condition["parameters"]
    associations = int(parameters["associations"])
    span_length = int(parameters["span_length"])
    query_keys_count = int(parameters["query_keys"])
    versions = int(parameters.get("versions", 1))
    keys = _unique_keys(
        generator,
        batch_size,
        SPAN_KEYS,
        associations,
    )
    values = generator.integers(
        0,
        SPAN_VALUES,
        size=(batch_size, versions, associations, span_length),
        dtype=np.uint16,
    )
    rows = np.arange(batch_size)[:, None]
    version_orders = np.argsort(
        generator.random((batch_size, versions, associations)),
        axis=2,
    )
    version_keys = np.take_along_axis(
        np.broadcast_to(keys[:, None, :], version_orders.shape),
        version_orders,
        axis=2,
    )
    version_values = np.take_along_axis(
        values,
        version_orders[:, :, :, None],
        axis=2,
    )
    slots = np.arange(span_length, dtype=np.uint32)
    mapping_offset = OVERWRITE_MAP_OFFSET if overwrite else SPAN_MAP_OFFSET
    mappings = (
        mapping_offset
        + (
            version_keys.astype(np.uint32)[:, :, :, None] * MAXIMUM_SPAN
            + slots
        )
        * SPAN_VALUES
        + version_values.astype(np.uint32)
    ).reshape(batch_size, -1)

    query_order = np.argsort(
        generator.random((batch_size, associations)),
        axis=1,
    )[:, :query_keys_count]
    query_keys = keys[rows, query_order]
    query_offset = OVERWRITE_QUERY_OFFSET if overwrite else SPAN_QUERY_OFFSET
    queries = (
        query_offset
        + query_keys.astype(np.uint32)[:, :, None] * MAXIMUM_SPAN
        + slots
    ).reshape(batch_size, -1)
    labels = values[
        np.arange(batch_size)[:, None],
        versions - 1,
        query_order,
    ].reshape(batch_size, -1)
    tokens = np.concatenate((mappings, queries), axis=1)
    return tokens.astype(np.uint32, copy=False), labels.astype(np.uint8)


def generate_batch(
    generator: np.random.Generator,
    batch_size: int,
    condition: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    family = str(condition["family"])
    if family == "pointer_chase":
        return _pointer_batch(generator, batch_size, condition)
    if family == "span_recall":
        return _span_batch(
            generator,
            batch_size,
            condition,
            overwrite=False,
        )
    if family == "overwrite_recall":
        return _span_batch(
            generator,
            batch_size,
            condition,
            overwrite=True,
        )
    raise ValueError(f"unknown condition family: {family}")


def _prepare_in_staging(
    output: Path,
    *,
    steps: int,
    batch_size: int,
    eval_examples: int,
    stream_seed: int,
    progress_every: int,
) -> Path:
    generator = np.random.default_rng(stream_seed)
    schedule = balanced_schedule(generator, len(CONDITIONS), steps)
    lengths = np.asarray(
        [int(CONDITIONS[int(index)]["sequence_length"]) for index in schedule],
        dtype=np.uint64,
    )
    train_offsets = np.empty(steps + 1, dtype=np.uint64)
    train_offsets[0] = 0
    np.cumsum(lengths * batch_size, out=train_offsets[1:])

    train_tokens_path = output / "train_tokens.uint32.bin"
    train_labels_path = output / "train_labels.uint8.bin"
    train_conditions_path = output / "train_condition_ids.uint16.bin"
    train_offsets_path = output / "train_token_offsets.uint64.bin"
    train_tokens = np.memmap(
        train_tokens_path,
        mode="w+",
        dtype=np.uint32,
        shape=(int(train_offsets[-1]),),
    )
    train_labels = np.memmap(
        train_labels_path,
        mode="w+",
        dtype=np.uint8,
        shape=(steps, batch_size, QUERIES),
    )
    for step, condition_index in enumerate(schedule):
        condition = CONDITIONS[int(condition_index)]
        tokens, labels = generate_batch(generator, batch_size, condition)
        start = int(train_offsets[step])
        stop = int(train_offsets[step + 1])
        train_tokens[start:stop] = tokens.reshape(-1)
        train_labels[step] = labels
        if progress_every > 0 and (step + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": "training_stream",
                        "completed_steps": step + 1,
                        "total_steps": steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    train_tokens.flush()
    train_labels.flush()
    del train_tokens, train_labels
    schedule.tofile(train_conditions_path)
    train_offsets.tofile(train_offsets_path)

    records: dict[str, dict[str, Any]] = {
        "train_tokens": record(
            output,
            train_tokens_path,
            "uint32",
            (int(train_offsets[-1]),),
        ),
        "train_labels": record(
            output,
            train_labels_path,
            "uint8",
            (steps, batch_size, QUERIES),
        ),
        "train_condition_ids": record(
            output,
            train_conditions_path,
            "uint16",
            (steps,),
        ),
        "train_token_offsets": record(
            output,
            train_offsets_path,
            "uint64",
            (steps + 1,),
        ),
    }

    for split, seed_offset in (
        ("validation", 10_000_000),
        ("test", 20_000_000),
    ):
        split_generator = np.random.default_rng(stream_seed + seed_offset)
        split_offsets = np.empty(len(CONDITIONS) + 1, dtype=np.uint64)
        split_offsets[0] = 0
        split_lengths = np.asarray(
            [
                int(condition["sequence_length"]) * eval_examples
                for condition in CONDITIONS
            ],
            dtype=np.uint64,
        )
        np.cumsum(split_lengths, out=split_offsets[1:])
        token_path = output / f"{split}_tokens.uint32.bin"
        label_path = output / f"{split}_labels.uint8.bin"
        offset_path = output / f"{split}_token_offsets.uint64.bin"
        split_tokens = np.memmap(
            token_path,
            mode="w+",
            dtype=np.uint32,
            shape=(int(split_offsets[-1]),),
        )
        split_labels = np.memmap(
            label_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(CONDITIONS), eval_examples, QUERIES),
        )
        for index, condition in enumerate(CONDITIONS):
            tokens, labels = generate_batch(
                split_generator,
                eval_examples,
                condition,
            )
            start = int(split_offsets[index])
            stop = int(split_offsets[index + 1])
            split_tokens[start:stop] = tokens.reshape(-1)
            split_labels[index] = labels
        split_tokens.flush()
        split_labels.flush()
        del split_tokens, split_labels
        split_offsets.tofile(offset_path)
        records[f"{split}_tokens"] = record(
            output,
            token_path,
            "uint32",
            (int(split_offsets[-1]),),
        )
        records[f"{split}_labels"] = record(
            output,
            label_path,
            "uint8",
            (len(CONDITIONS), eval_examples, QUERIES),
        )
        records[f"{split}_token_offsets"] = record(
            output,
            offset_path,
            "uint64",
            (len(CONDITIONS) + 1,),
        )

    manifest = {
        "format": FORMAT,
        "task": "adaptive_depth_exact_recall_suite",
        "steps": steps,
        "batch_size": batch_size,
        "eval_examples": eval_examples,
        "stream_seed": stream_seed,
        "maximum_queries": QUERIES,
        "maximum_sequence_length": max(
            int(row["sequence_length"]) for row in CONDITIONS
        ),
        "output_classes": OUTPUT_CLASSES,
        "vocab_size": VOCAB_SIZE,
        "conditions": list(CONDITIONS),
        "families": {
            "pointer_chase": {
                "definition": (
                    "follow a random permutation-valued key map for a "
                    "query-specified dependent hop count"
                ),
                "key_vocabulary": POINTER_KEYS,
                "hop_depths": list(POINTER_HOPS),
                "association_counts": [32, 64, 128, 192],
            },
            "span_recall": {
                "definition": (
                    "retrieve every symbol of random multi-token values using "
                    "distinct key-and-slot queries"
                ),
                "key_vocabulary": SPAN_KEYS,
                "value_classes": SPAN_VALUES,
                "association_counts": [16, 32],
                "span_lengths": [1, 4, 8, 16],
            },
            "overwrite_recall": {
                "definition": (
                    "retrieve the latest random value after repeated updates "
                    "to the same key-and-slot identities"
                ),
                "key_vocabulary": SPAN_KEYS,
                "value_classes": SPAN_VALUES,
                "association_counts": [16, 32],
                "versions": [1, 2, 4],
                "span_length": 4,
            },
        },
        "token_codec": {
            "pointer_mapping": {
                "offset": POINTER_MAP_OFFSET,
                "count": POINTER_MAP_TOKENS,
                "formula": "offset + source_key * pointer_keys + target_key",
            },
            "pointer_query": {
                "offset": POINTER_QUERY_OFFSET,
                "count": POINTER_QUERY_TOKENS,
                "formula": "offset + hop_index * pointer_keys + start_key",
            },
            "span_mapping": {
                "offset": SPAN_MAP_OFFSET,
                "count": SPAN_MAP_TOKENS,
                "formula": (
                    "offset + (key * maximum_span + slot) * "
                    "span_values + value"
                ),
            },
            "span_query": {
                "offset": SPAN_QUERY_OFFSET,
                "count": SPAN_QUERY_TOKENS,
                "formula": "offset + key * maximum_span + slot",
            },
            "overwrite_mapping": {
                "offset": OVERWRITE_MAP_OFFSET,
                "count": OVERWRITE_MAP_TOKENS,
                "formula": (
                    "offset + (key * maximum_span + slot) * "
                    "span_values + value"
                ),
            },
            "overwrite_query": {
                "offset": OVERWRITE_QUERY_OFFSET,
                "count": OVERWRITE_QUERY_TOKENS,
                "formula": "offset + key * maximum_span + slot",
            },
        },
        "records": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def prepare_dataset(
    output: Path,
    *,
    steps: int,
    batch_size: int,
    eval_examples: int,
    stream_seed: int,
    progress_every: int = 1_000,
) -> Path:
    output = output.resolve()
    if output.exists():
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            requested = {
                "steps": steps,
                "batch_size": batch_size,
                "eval_examples": eval_examples,
                "stream_seed": stream_seed,
            }
            if manifest.get("format") == FORMAT and all(
                int(manifest.get(name, -1)) == value
                for name, value in requested.items()
            ):
                AdaptiveDepthData(manifest_path)
                return manifest_path
        if any(output.iterdir()):
            raise ValueError(
                f"output directory contains a different dataset: {output}"
            )
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            dir=output.parent,
        )
    )
    try:
        manifest = _prepare_in_staging(
            staging,
            steps=steps,
            batch_size=batch_size,
            eval_examples=eval_examples,
            stream_seed=stream_seed,
            progress_every=progress_every,
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / manifest.name


class AdaptiveDepthData:
    """Hash-verified reader for the immutable ragged multitask stream."""

    _DTYPES = {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    }

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT:
            raise ValueError("unexpected adaptive-depth manifest format")
        self.conditions = tuple(self.manifest["conditions"])
        self.condition_by_id = {
            str(row["id"]): row for row in self.conditions
        }
        if len(self.condition_by_id) != len(self.conditions):
            raise ValueError("condition identifiers are not unique")
        self._arrays: dict[str, np.memmap] = {}
        for name, row in self.manifest["records"].items():
            path = self.root / str(row["path"])
            if path.stat().st_size != int(row["bytes"]):
                raise ValueError(f"size mismatch for {path}")
            digest = sha256_file(path)
            if digest != str(row["sha256"]):
                raise ValueError(
                    f"SHA-256 mismatch for {path}: {digest} != {row['sha256']}"
                )
            dtype_name = str(row["dtype"])
            if dtype_name not in self._DTYPES:
                raise ValueError(f"unsupported dtype {dtype_name!r}")
            self._arrays[name] = np.memmap(
                path,
                mode="r",
                dtype=self._DTYPES[dtype_name],
                shape=tuple(int(value) for value in row["shape"]),
            )
        self.train_tokens = self._arrays["train_tokens"]
        self.train_labels = self._arrays["train_labels"]
        self.train_condition_ids = self._arrays["train_condition_ids"]
        self.train_offsets = self._arrays["train_token_offsets"]

    def condition(self, identifier: str | int) -> dict[str, Any]:
        if isinstance(identifier, str):
            if identifier not in self.condition_by_id:
                raise ValueError(f"unknown condition: {identifier}")
            return self.condition_by_id[identifier]
        index = int(identifier)
        if index < 0 or index >= len(self.conditions):
            raise ValueError(f"condition index out of range: {index}")
        return self.conditions[index]

    def train_batch(
        self,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        condition = self.condition(int(self.train_condition_ids[step]))
        length = int(condition["sequence_length"])
        start = int(self.train_offsets[step])
        stop = int(self.train_offsets[step + 1])
        tokens = np.asarray(self.train_tokens[start:stop]).reshape(
            int(self.manifest["batch_size"]),
            length,
        )
        labels = np.asarray(self.train_labels[step])
        return tokens, labels, condition

    def evaluation(
        self,
        split: str,
        identifier: str | int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if split not in ("validation", "test"):
            raise ValueError(f"unsupported split: {split}")
        condition = self.condition(identifier)
        index = int(condition["index"])
        offsets = self._arrays[f"{split}_token_offsets"]
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        tokens = np.asarray(self._arrays[f"{split}_tokens"][start:stop]).reshape(
            int(self.manifest["eval_examples"]),
            int(condition["sequence_length"]),
        )
        labels = np.asarray(self._arrays[f"{split}_labels"][index])
        return tokens, labels

    def _verify_pointer(
        self,
        tokens: np.ndarray,
        labels: np.ndarray,
        condition: dict[str, Any],
    ) -> None:
        associations = int(condition["parameters"]["associations"])
        hops = int(condition["parameters"]["hops"])
        mappings = tokens[:, :associations] - POINTER_MAP_OFFSET
        sources = mappings // POINTER_KEYS
        targets = mappings % POINTER_KEYS
        if np.any(
            np.sort(sources, axis=1)[:, 1:]
            == np.sort(sources, axis=1)[:, :-1]
        ):
            raise ValueError("pointer source keys are not unique")
        if not np.array_equal(
            np.sort(sources, axis=1),
            np.sort(targets, axis=1),
        ):
            raise ValueError("pointer targets are not a permutation of sources")
        queries = tokens[:, associations:] - POINTER_QUERY_OFFSET
        hop_indices = queries // POINTER_KEYS
        expected_hop_index = POINTER_HOPS.index(hops)
        if np.any(hop_indices != expected_hop_index):
            raise ValueError("pointer query hop encoding mismatch")
        starts = queries % POINTER_KEYS
        rows = np.arange(len(tokens))[:, None]
        table = np.zeros((len(tokens), POINTER_KEYS), dtype=np.uint16)
        table[rows, sources] = targets
        walked = sources.astype(np.uint16)
        for _ in range(max(POINTER_HOPS)):
            walked = table[rows, walked]
            if np.any(walked == sources):
                raise ValueError("pointer map has a cycle shorter than maximum hops")
        expected = starts.astype(np.uint16)
        for _ in range(hops):
            expected = table[rows, expected]
        if not np.array_equal(labels, expected.astype(np.uint8)):
            raise ValueError("pointer labels do not match the requested chain")

    def _verify_span(
        self,
        tokens: np.ndarray,
        labels: np.ndarray,
        condition: dict[str, Any],
        *,
        overwrite: bool,
    ) -> None:
        parameters = condition["parameters"]
        associations = int(parameters["associations"])
        span_length = int(parameters["span_length"])
        versions = int(parameters.get("versions", 1))
        memory_tokens = int(condition["memory_tokens"])
        mapping_offset = OVERWRITE_MAP_OFFSET if overwrite else SPAN_MAP_OFFSET
        query_offset = OVERWRITE_QUERY_OFFSET if overwrite else SPAN_QUERY_OFFSET
        local = tokens[:, :memory_tokens] - mapping_offset
        values = local % SPAN_VALUES
        key_slots = local // SPAN_VALUES
        slots = key_slots % MAXIMUM_SPAN
        keys = key_slots // MAXIMUM_SPAN
        shaped_keys = keys.reshape(
            len(tokens),
            versions,
            associations,
            span_length,
        )
        shaped_slots = slots.reshape(
            len(tokens),
            versions,
            associations,
            span_length,
        )
        expected_slots = np.arange(span_length)[None, None, None, :]
        if np.any(shaped_slots != expected_slots):
            raise ValueError("mapping slots are not ordered within each span")
        if np.any(shaped_keys != shaped_keys[..., :1]):
            raise ValueError("mapping key changes within a span")
        for version in range(versions):
            version_keys = shaped_keys[:, version, :, 0]
            if np.any(
                np.sort(version_keys, axis=1)[:, 1:]
                == np.sort(version_keys, axis=1)[:, :-1]
            ):
                raise ValueError("mapping keys are not unique within a version")

        query_local = tokens[:, memory_tokens:] - query_offset
        query_slots = query_local % MAXIMUM_SPAN
        query_keys = query_local // MAXIMUM_SPAN
        query_groups = int(condition["groups"])
        shaped_query_keys = query_keys.reshape(
            len(tokens),
            query_groups,
            span_length,
        )
        shaped_query_slots = query_slots.reshape(
            len(tokens),
            query_groups,
            span_length,
        )
        if np.any(shaped_query_slots != np.arange(span_length)[None, None, :]):
            raise ValueError("query slots are not ordered within each span")
        if np.any(shaped_query_keys != shaped_query_keys[..., :1]):
            raise ValueError("query key changes within a span")

        last_keys = shaped_keys[:, -1]
        last_values = values.reshape(
            len(tokens),
            versions,
            associations,
            span_length,
        )[:, -1]
        value_table = np.zeros(
            (len(tokens), SPAN_KEYS, span_length),
            dtype=np.uint8,
        )
        rows = np.arange(len(tokens))[:, None, None]
        value_table[
            rows,
            last_keys,
            np.arange(span_length)[None, None, :],
        ] = last_values.astype(np.uint8)
        expected = value_table[
            np.arange(len(tokens))[:, None, None],
            shaped_query_keys,
            shaped_query_slots,
        ].reshape(len(tokens), QUERIES)
        if not np.array_equal(labels, expected):
            raise ValueError("span labels do not match the latest mapped values")

    def _verify_batch(
        self,
        tokens: np.ndarray,
        labels: np.ndarray,
        condition: dict[str, Any],
    ) -> None:
        expected_shape = (
            len(tokens),
            int(condition["sequence_length"]),
        )
        if tokens.shape != expected_shape:
            raise ValueError(f"token shape {tokens.shape} != {expected_shape}")
        if labels.shape != (len(tokens), QUERIES):
            raise ValueError("label shape does not match fixed query count")
        if np.any(tokens >= int(self.manifest["vocab_size"])):
            raise ValueError("token exceeds declared vocabulary")
        family = str(condition["family"])
        if family == "pointer_chase":
            self._verify_pointer(tokens, labels, condition)
        elif family == "span_recall":
            self._verify_span(
                tokens,
                labels,
                condition,
                overwrite=False,
            )
        elif family == "overwrite_recall":
            self._verify_span(
                tokens,
                labels,
                condition,
                overwrite=True,
            )
        else:
            raise ValueError(f"unsupported family: {family}")

    def verify_semantics(self) -> None:
        counts, frequencies = np.unique(
            np.asarray(self.train_condition_ids),
            return_counts=True,
        )
        if tuple(int(value) for value in counts) != tuple(range(len(self.conditions))):
            raise ValueError("training schedule omits conditions")
        if int(frequencies.max() - frequencies.min()) > 1:
            raise ValueError("training schedule is not balanced")
        steps = int(self.manifest["steps"])
        sample_steps = np.linspace(
            0,
            steps - 1,
            num=min(64, steps),
            dtype=int,
        )
        for step in sample_steps:
            tokens, labels, condition = self.train_batch(int(step))
            self._verify_batch(tokens, labels, condition)
        for split in ("validation", "test"):
            for condition in self.conditions:
                tokens, labels = self.evaluation(split, int(condition["index"]))
                self._verify_batch(tokens, labels, condition)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-examples", type=int, default=2_048)
    parser.add_argument("--stream-seed", type=int, default=102_337)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "eval_examples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps < len(CONDITIONS):
        parser.error(
            f"--steps must be at least the {len(CONDITIONS)} condition count"
        )
    return args


def main() -> None:
    args = parse_args()
    manifest_path = prepare_dataset(
        args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_examples=args.eval_examples,
        stream_seed=args.stream_seed,
        progress_every=args.progress_every,
    )
    data = AdaptiveDepthData(manifest_path)
    data.verify_semantics()
    manifest_sha256 = sha256_file(manifest_path)
    if (
        args.expected_manifest_sha256 is not None
        and manifest_sha256 != args.expected_manifest_sha256
    ):
        raise ValueError(
            "prepared recall manifest identity changed: "
            f"{manifest_sha256} != {args.expected_manifest_sha256}"
        )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "payload_bytes": sum(
                    int(row["bytes"])
                    for row in data.manifest["records"].values()
                ),
                "conditions": len(data.conditions),
                "semantic_verification": "passed",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
