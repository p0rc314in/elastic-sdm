from __future__ import annotations

from pathlib import Path
import runpy
import unittest

import torch

from benchmarks.train_wikitext_cuda import route_occupancy_counts


ROOT = Path(__file__).resolve().parents[1]


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class ReproductionDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = runpy.run_path(ROOT / "scripts/run_experiments.py")

    def test_release_has_only_nine_balanced_write_arms(self) -> None:
        recall = self.module["RECALL_ARMS"]
        language = self.module["LANGUAGE_ARMS"]
        self.assertEqual(len(recall), 7)
        self.assertEqual(len(language), 2)
        self.assertEqual(
            [row[2] for row in recall],
            [0.0, 0.012, 0.024, 0.04, 0.12, 0.24, 0.4],
        )
        self.assertEqual([row[2] for row in language], [0.0, 0.024])
        self.assertTrue(all(row[1] == 1_024 for row in recall + language))

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
            self.assertEqual(option(command, "--slots"), "1024")
            self.assertEqual(option(command, "--reads"), "16")
            self.assertEqual(option(command, "--writes"), "16")
            self.assertEqual(option(command, "--seed"), "0")

    def test_standalone_trainers_default_to_balanced_writes(self) -> None:
        for relative in (
            "benchmarks/train_recall_cuda.py",
            "benchmarks/train_wikitext_cuda.py",
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
