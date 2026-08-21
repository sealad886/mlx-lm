#!/usr/bin/env python3
import unittest
from types import SimpleNamespace

from mlx_lm.quant.dwq import extract_logits, save_interrupt_checkpoint


class TestExtractLogits(unittest.TestCase):
    def test_preserves_raw_logits(self) -> None:
        logits = object()

        self.assertIs(logits, extract_logits(logits))

    def test_extracts_structured_model_output_logits(self) -> None:
        logits = object()

        self.assertIs(logits, extract_logits(SimpleNamespace(logits=logits)))

    def test_rejects_null_structured_logits(self) -> None:
        with self.assertRaisesRegex(TypeError, "null 'logits'"):
            extract_logits(SimpleNamespace(logits=None))


class TestInterruptCheckpoint(unittest.TestCase):
    def test_forces_checkpoint_for_last_completed_iteration(self) -> None:
        calls = []

        class FakeMX:
            @staticmethod
            def eval(*values):
                calls.append(("eval", values))

        def checkpoint(*args, **kwargs):
            calls.append(("checkpoint", args, kwargs))

        saved = save_interrupt_checkpoint(
            checkpoint,
            mx=FakeMX,
            completed_iterations=5,
            start_iteration=0,
            params="params",
            optimizer_state="optimizer",
            initial_valid_loss=0.9,
            valid_loss=0.9,
        )

        self.assertTrue(saved)
        self.assertEqual(("eval", ("params", "optimizer")), calls[0])
        self.assertEqual(5, calls[1][1][0])
        self.assertEqual({"force": True}, calls[1][2])

    def test_skips_checkpoint_without_new_completed_iteration(self) -> None:
        saved = save_interrupt_checkpoint(
            lambda *_args, **_kwargs: self.fail("checkpoint should not run"),
            mx=SimpleNamespace(eval=lambda *_args: None),
            completed_iterations=5,
            start_iteration=5,
            params="params",
            optimizer_state="optimizer",
            initial_valid_loss=0.9,
            valid_loss=0.9,
        )

        self.assertFalse(saved)


if __name__ == "__main__":
    unittest.main()
