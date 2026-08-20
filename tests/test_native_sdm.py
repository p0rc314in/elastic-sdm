from __future__ import annotations

import copy
import unittest

import torch

from elastic_sdm.model import LanguageModel
from elastic_sdm.sdm import (
    SparseDeltaMemory,
    _released_fused_chunk_size,
    dense_recurrence_block_width,
    dense_recurrence_slot_tiles,
)


class NativeSDMTest(unittest.TestCase):
    def test_released_fused_chunks_are_power_of_two_and_bounded(self) -> None:
        expected = {
            1: 16,
            15: 16,
            16: 16,
            31: 16,
            48: 32,
            208: 128,
            528: 256,
        }
        for time, chunk in expected.items():
            self.assertEqual(_released_fused_chunk_size(time), chunk)
            self.assertEqual(chunk & (chunk - 1), 0)
            self.assertLessEqual(chunk, 256)
        with self.assertRaisesRegex(ValueError, "time must be positive"):
            _released_fused_chunk_size(0)

    def test_dense_recurrence_tile_stays_bounded_through_n4096(self) -> None:
        expected = {
            16: 16,
            256: 16,
            512: 8,
            1_024: 4,
            2_304: 2,
            4_096: 1,
        }
        for slots, width in expected.items():
            self.assertEqual(dense_recurrence_block_width(slots), width)
            main_slots, tail_slots = dense_recurrence_slot_tiles(slots)
            self.assertLessEqual((main_slots + tail_slots) * width, 4_608)
        with self.assertRaisesRegex(ValueError, "at most 4,096"):
            dense_recurrence_block_width(4_097)
        with self.assertRaisesRegex(ValueError, "at most 4,096"):
            dense_recurrence_slot_tiles(4_097)

    def test_dense_recurrence_slot_tiles_avoid_n2304_padding(self) -> None:
        self.assertEqual(dense_recurrence_slot_tiles(1_024), (1_024, 0))
        self.assertEqual(dense_recurrence_slot_tiles(2_304), (2_048, 256))
        self.assertEqual(dense_recurrence_slot_tiles(4_096), (4_096, 0))

    def test_n2304_controller_preserves_serial_prefix_causality(self) -> None:
        torch.manual_seed(11)
        module = SparseDeltaMemory(16, slots=2_304, reads=4, writes=4).eval()
        tokens = torch.randn(1, 4, 16)
        changed = tokens.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:])
        with torch.no_grad():
            baseline = module(tokens, serial_reference=True)
            counterfactual = module(changed, serial_reference=True)
        self.assertEqual(
            float((baseline[:, :3] - counterfactual[:, :3]).abs().max()),
            0.0,
        )

    def test_n1024_controller_preserves_serial_prefix_causality(self) -> None:
        torch.manual_seed(10)
        module = SparseDeltaMemory(16, slots=1_024, reads=16, writes=16).eval()
        tokens = torch.randn(1, 4, 16)
        changed = tokens.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:])
        with torch.no_grad():
            baseline = module(tokens, serial_reference=True)
            counterfactual = module(changed, serial_reference=True)
        self.assertEqual(
            float((baseline[:, :3] - counterfactual[:, :3]).abs().max()),
            0.0,
        )

    def test_balanced_n1024_model_accounting(self) -> None:
        model = LanguageModel(
            vocab_size=50_257,
            maximum_sequence_length=2_048,
            layout="BBBBBBBA",
            width=128,
            heads=4,
            slots=1_024,
            reads=16,
            writes=16,
            memory_heads=1,
            mlp_expansion=4.0,
            activation_dtype=torch.float32,
        )
        model.initialize_role_keyed(0)
        learned = sum(
            parameter.numel()
            for parameter in model.parameters()
            if getattr(parameter, "_sdm_memory_bank", False)
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(learned, 917_504)
        self.assertEqual(total - learned, 14_712_348)

    def test_optimized_interface_matches_serial_outputs_and_gradients(self) -> None:
        torch.manual_seed(7)
        fast = SparseDeltaMemory(16, slots=16, reads=16, writes=16)
        reference = copy.deepcopy(fast)
        source = torch.randn(2, 7, 16, requires_grad=True)
        reference_source = source.detach().clone().requires_grad_(True)
        probe = torch.randn(2, 7, 16)

        output = fast(source)
        serial_output = reference(reference_source, serial_reference=True)
        (output * probe).sum().backward()
        (serial_output * probe).sum().backward()

        self.assertTrue(torch.equal(output, serial_output))
        self.assertTrue(torch.equal(source.grad, reference_source.grad))
        for (name, parameter), (reference_name, reference_parameter) in zip(
            fast.named_parameters(), reference.named_parameters(), strict=True
        ):
            self.assertEqual(name, reference_name)
            self.assertTrue(
                torch.equal(parameter.grad, reference_parameter.grad), msg=name
            )

    def test_prefix_causality_is_exact(self) -> None:
        torch.manual_seed(9)
        module = SparseDeltaMemory(16, slots=16, reads=16, writes=16).eval()
        tokens = torch.randn(2, 9, 16)
        changed = tokens.clone()
        changed[:, 5:] = torch.randn_like(changed[:, 5:])
        with torch.no_grad():
            baseline = module(tokens)
            counterfactual = module(changed)
        self.assertEqual(
            float((baseline[:, :5] - counterfactual[:, :5]).abs().max()),
            0.0,
        )

    def test_controller_has_only_native_projection_path(self) -> None:
        module = SparseDeltaMemory(16, slots=16, reads=16, writes=16)
        names = {name for name, _ in module.named_parameters()}
        expected_prefixes = {
            "read_projection",
            "write_projection",
            "value_projection",
            "forget_projection",
            "input_projection",
            "output_gate",
            "output_projection",
        }
        for prefix in expected_prefixes:
            self.assertTrue(any(name.startswith(prefix) for name in names), prefix)
        self.assertIn("initial_memory", names)


if __name__ == "__main__":
    unittest.main()
