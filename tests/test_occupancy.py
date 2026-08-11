from __future__ import annotations

import unittest

import torch

from elastic_sdm.sdm import SDMRouting, sdm_copy_on_write_accounting


def routing(indices: torch.Tensor, logits: torch.Tensor) -> SDMRouting:
    weights = torch.softmax(logits, dim=-1)
    batch, time, heads, _ = indices.shape
    reads = indices[..., :1]
    return SDMRouting(
        read_indices=reads,
        read_weights=torch.ones(batch, time, heads, 1),
        write_indices=indices,
        write_weights=weights,
        forget_log_gate=torch.zeros(batch, time, heads),
        erase_gate=torch.zeros(batch, time, heads),
        input_gate=torch.ones(batch, time, heads),
    )


class OccupancyObjectiveTest(unittest.TestCase):
    def test_hard_forward_counts_unique_top_w_addresses(self) -> None:
        indices = torch.tensor([[[[0, 1]], [[1, 2]], [[2, 3]]]])
        logits = torch.zeros(1, 3, 1, 2, requires_grad=True)
        result = sdm_copy_on_write_accounting(routing(indices, logits), slots=8)
        self.assertEqual(result.hard_unique_by_position.tolist(), [[ [2], [3], [4] ]])
        self.assertEqual(float(result.hard_final_fraction), 0.5)
        self.assertEqual(result.first_touch_count_by_position.tolist(), [[[2], [1], [1]]])
        self.assertEqual(result.repeated_write_count_by_position.tolist(), [[[0], [1], [1]]])

    def test_straight_through_value_is_hard_and_gradient_reaches_router(self) -> None:
        indices = torch.tensor([[[[0, 1]], [[0, 1]], [[0, 1]]]])
        logits = torch.tensor(
            [[[[0.2, -0.2]], [[-0.1, 0.1]], [[0.4, -0.4]]]],
            requires_grad=True,
        )
        result = sdm_copy_on_write_accounting(routing(indices, logits), slots=8)
        self.assertEqual(
            float(result.straight_through_fraction.detach()),
            float(result.hard_final_fraction),
        )
        result.straight_through_fraction.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_single_position_has_no_route_distribution_gradient(self) -> None:
        indices = torch.tensor([[[[0, 1, 2, 3]]]])
        logits = torch.randn(1, 1, 1, 4, requires_grad=True)
        result = sdm_copy_on_write_accounting(routing(indices, logits), slots=8)
        result.straight_through_fraction.backward()
        self.assertLess(float(logits.grad.abs().max()), 1e-7)


if __name__ == "__main__":
    unittest.main()
