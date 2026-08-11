from __future__ import annotations

import unittest

import torch

from elastic_sdm.allocator import (
    PackedSDMCopyOnWriteState,
    dense_sdm_sparse_step,
    packed_sdm_cow_step,
)


class PackedAllocatorTest(unittest.TestCase):
    def test_balanced_w16_matches_dense_state_and_reads(self) -> None:
        torch.manual_seed(31)
        banks, slots, width = 2, 64, 16
        initial = torch.randn(1, slots, width)
        dense = initial.expand(banks, -1, -1).clone()
        packed = PackedSDMCopyOnWriteState.allocate(
            initial,
            banks=banks,
            capacity_rows=banks * slots,
            state_dtype=torch.float32,
        )
        for step in range(3):
            writes = torch.stack(
                [
                    (torch.arange(16) + row + step) % slots
                    for row in range(banks)
                ]
            )
            reads = torch.stack(
                [(torch.arange(16) + 2 * row) % slots for row in range(banks)]
            )
            write_weights = torch.softmax(torch.randn(banks, 16), dim=-1)
            read_weights = torch.softmax(torch.randn(banks, 16), dim=-1)
            values = torch.randn(banks, width)
            input_gate = torch.sigmoid(torch.randn(banks))
            forget = -torch.rand(banks)
            dense_read = dense_sdm_sparse_step(
                dense,
                writes,
                write_weights,
                values,
                input_gate,
                forget,
                reads,
                read_weights,
            )
            packed_read, _ = packed_sdm_cow_step(
                packed,
                writes,
                write_weights,
                values,
                input_gate,
                forget,
                reads,
                read_weights,
                backend="torch",
            )
            self.assertTrue(torch.equal(packed_read, dense_read))
            self.assertTrue(torch.equal(packed.materialize_dense(), dense))
        self.assertTrue(packed.validate_invariants()["exact_partition"])


if __name__ == "__main__":
    unittest.main()
