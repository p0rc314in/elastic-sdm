from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import unittest

import torch

from benchmarks.wikitext_support import route_occupancy_counts
from benchmarks.wikitext103_coverage import (
    OPTIMIZER_STEPS_PER_PASS,
    STREAM_SEED,
    TRAIN_TARGETS_PER_PASS,
    build_pass_permutations,
    build_training_index,
    validate_training_index,
)


ROOT = Path(__file__).resolve().parents[1]


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class ReproductionDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = runpy.run_path(ROOT / "scripts/run_experiments.py")

    def test_release_has_thirteen_unique_decision_bearing_arms(self) -> None:
        recall = self.module["RECALL_ARMS"]
        language = self.module["LANGUAGE_ARMS"]
        self.assertEqual(len(recall), 8)
        self.assertEqual(len(language), 5)
        self.assertEqual(
            [(row[1], row[2]) for row in recall],
            [
                (1_024, 0.0),
                (1_024, 0.024),
                (1_024, 0.12),
                (1_024, 0.24),
                (1_024, 0.40),
                (2_304, 0.0),
                (2_304, 0.054),
                (2_304, 0.27),
            ],
        )
        self.assertEqual([row[3] for row in recall], [0] * 8)
        self.assertEqual(
            [row[2] for row in language],
            [0.0, 0.024, 0.12, 0.24, 0.4],
        )
        self.assertTrue(all(row[1] == 1_024 for row in language))

    def test_every_command_uses_one_memory_head_and_r_equals_w_equals_16(self) -> None:
        recall_command = self.module["recall_command"]
        language_command = self.module["language_command"]
        commands = [
            recall_command(Path("manifest"), Path("output"), arm)
            for arm in self.module["RECALL_ARMS"]
        ] + [
            language_command(Path("manifest"), Path("output"), arm)
            for arm in self.module["LANGUAGE_ARMS"]
        ]
        for command in commands:
            self.assertEqual(option(command, "--memory-heads"), "1")
            self.assertEqual(option(command, "--reads"), "16")
            self.assertEqual(option(command, "--writes"), "16")
        recall_commands = commands[: len(self.module["RECALL_ARMS"])]
        language_commands = commands[len(self.module["RECALL_ARMS"]) :]
        self.assertEqual(
            [int(option(command, "--slots")) for command in recall_commands],
            [1_024, 1_024, 1_024, 1_024, 1_024, 2_304, 2_304, 2_304],
        )
        self.assertTrue(
            all(option(command, "--seed") == "0" for command in recall_commands)
        )
        self.assertTrue(
            all(option(command, "--slots") == "1024" for command in language_commands)
        )
        self.assertTrue(
            all(option(command, "--seed") == "0" for command in language_commands)
        )

    def test_language_commands_use_complete_canonical_coverage(self) -> None:
        commands = [
            self.module["language_command"](Path("manifest"), Path("output"), arm)
            for arm in self.module["LANGUAGE_ARMS"]
        ]
        for command in commands:
            self.assertEqual(
                command[2],
                "benchmarks.train_wikitext_coverage_cuda",
            )
            self.assertEqual(option(command, "--passes"), "3")
            self.assertEqual(option(command, "--steps"), "21603")
            self.assertEqual(option(command, "--schedule-steps"), "21603")
            self.assertEqual(option(command, "--warmup-steps"), "540")
            self.assertEqual(option(command, "--stream-seed"), "20260818")
            self.assertEqual(
                option(command, "--expected-manifest-sha256"),
                "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6",
            )

    def test_canonical_wikitext_index_covers_three_complete_passes(self) -> None:
        starts, targets = build_training_index(117_980_450)
        permutations = build_pass_permutations(len(starts), 3, STREAM_SEED)
        coverage = validate_training_index(
            starts,
            targets,
            permutations,
            token_count=117_980_450,
        )
        self.assertEqual(len(starts), 57_608)
        self.assertEqual(OPTIMIZER_STEPS_PER_PASS * 3, 21_603)
        self.assertEqual(coverage["targets_per_pass"], TRAIN_TARGETS_PER_PASS)
        self.assertEqual(coverage["total_target_presentations"], 353_941_347)

    def test_standalone_trainers_default_to_balanced_writes(self) -> None:
        for relative in (
            "benchmarks/train_recall_cuda.py",
            "benchmarks/train_wikitext_coverage_cuda.py",
        ):
            source = (ROOT / relative).read_text()
            self.assertIn(
                'parser.add_argument("--writes", type=int, default=16)',
                source,
            )
            self.assertNotIn(
                'parser.add_argument("--writes", type=int, default=4)',
                source,
            )

    def test_wikitext_preparation_keeps_generated_data_out_of_source_data(
        self,
    ) -> None:
        source = (ROOT / "scripts/prepare_wikitext.py").read_text()
        self.assertIn(
            'default=ROOT / "runs/data/wikitext103_gpt2_causal_t2048_coverage_v1"',
            source,
        )
        self.assertIn(
            'default=ROOT / "runs/data/downloads/wikitext103"',
            source,
        )
        self.assertIn('"TIKTOKEN_CACHE_DIR"', source)

    def test_released_sdm_source_is_the_recorded_unmodified_tree(self) -> None:
        source_root = ROOT / "third_party/released_sdm"
        manifest = json.loads((source_root / "SOURCE.json").read_text())
        self.assertFalse(manifest["modified"])
        observed = {
            str(path.relative_to(source_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((source_root / "lingua").rglob("*"))
            if path.is_file()
        }
        self.assertEqual(observed, manifest["files"])
        license_record = manifest["license_file"]
        license_path = source_root / license_record["path"]
        self.assertEqual(
            hashlib.sha256(license_path.read_bytes()).hexdigest(),
            license_record["sha256"],
        )

    def test_reproduction_requires_the_released_fused_cuda_path(self) -> None:
        source = (ROOT / "scripts/run_experiments.py").read_text()
        self.assertIn(
            'environment["ELASTIC_SDM_REQUIRE_RELEASED_FUSED"] = "1"',
            source,
        )
        project = (ROOT / "pyproject.toml").read_text()
        self.assertIn('torch==2.11.0', project)
        self.assertIn('triton==3.6.0', project)
        self.assertIn('pyarrow==25.0.1', project)
        self.assertIn('tiktoken==0.14.0', project)
        self.assertIn('url = "https://download.pytorch.org/whl/cu128"', project)
        self.assertIn('"third_party/released_sdm/LICENSE"', project)
        self.assertNotIn('torch==2.8.0', project)

    def test_batched_occupancy_matches_per_head_unique_counts(self) -> None:
        torch.manual_seed(41)
        routes = torch.randint(0, 8, (3, 2, 5, 2, 4), dtype=torch.int16)
        totals, by_layer = route_occupancy_counts(routes, prefix=4, slots=8)
        expected = torch.tensor(
            [
                [
                    sum(
                        torch.unique(routes[example, layer, :4, head]).numel()
                        for head in range(2)
                    )
                    for layer in range(2)
                ]
                for example in range(3)
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(by_layer, expected))
        self.assertTrue(torch.equal(totals, expected.sum(dim=1)))


if __name__ == "__main__":
    unittest.main()
