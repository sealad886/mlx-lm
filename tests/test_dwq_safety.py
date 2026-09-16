import inspect
import unittest
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn

import mlx_lm.quant.dwq as dwq


class TestDWQSafety(unittest.TestCase):
    def test_dwq_quantize_default_validation_interval_matches_legacy_cadence(self):
        default = inspect.signature(dwq.dwq_quantize).parameters[
            "validation_interval"
        ].default
        self.assertEqual(200, default)

    def test_checkpoint_due_gates_validation_cadence(self):
        # dwq_quantize reuses checkpoint_due to decide when a full
        # validation pass is due during training.
        self.assertFalse(dwq.checkpoint_due(199, 200))
        self.assertTrue(dwq.checkpoint_due(200, 200))
        self.assertFalse(dwq.checkpoint_due(200, 0))

    def test_checkpoint_due_uses_completed_iteration_interval(self):
        due = getattr(dwq, "checkpoint_due", None)
        self.assertTrue(callable(due), "DWQ checkpoint cadence helper is missing")
        self.assertFalse(due(19, 20))
        self.assertTrue(due(20, 20))
        self.assertFalse(due(20, 0))

    def test_checkpoint_interval_one_is_due_after_every_iteration(self):
        self.assertTrue(
            all(dwq.checkpoint_due(iteration, 1) for iteration in range(1, 6))
        )

    def test_final_checkpoint_due_honors_disabled_interval(self):
        due = getattr(dwq, "final_checkpoint_due", None)
        self.assertTrue(callable(due), "DWQ final checkpoint helper is missing")
        self.assertFalse(due(20, 0))
        self.assertFalse(due(20, 20))
        self.assertTrue(due(21, 20))

    def test_dwq_event_callback_receives_kind_and_fields(self):
        events = []
        dwq.emit_dwq_event(
            lambda kind, fields: events.append((kind, fields)),
            "training_iteration",
            iteration=9,
            loss=0.125,
        )
        self.assertEqual(
            [("training_iteration", {"iteration": 9, "loss": 0.125})],
            events,
        )

    def test_dwq_event_callback_is_optional(self):
        self.assertIsNone(dwq.emit_dwq_event(None, "validation_start", iteration=0))

    def test_nonfinite_loss_emits_guard_event_before_rejection(self):
        guard = getattr(dwq, "ensure_finite", None)
        self.assertTrue(callable(guard), "DWQ finite guard is missing")
        events = []
        try:
            with self.assertRaisesRegex(FloatingPointError, "iteration 9.*loss"):
                guard(
                    mx.array(float("nan")),
                    label="loss",
                    iteration=9,
                    event_fn=lambda kind, fields: events.append((kind, fields)),
                )
        except TypeError as exc:
            self.fail(f"DWQ finite guard event callback is unavailable: {exc}")
        self.assertEqual(
            [
                (
                    "nonfinite_loss",
                    {
                        "iteration": 9,
                        "label": "loss",
                        "message": "DWQ iteration 9 produced non-finite loss; aborting",
                    },
                )
            ],
            events,
        )

    def test_trainable_parameter_report_emits_exact_structured_metric(self):
        report = getattr(dwq, "report_trainable_parameters", None)
        self.assertTrue(callable(report), "DWQ trainable-parameter reporter is missing")
        model = MagicMock()
        leaf = MagicMock(spec=nn.Linear)
        leaf.weight = MagicMock(size=10)
        leaf.parameters.return_value = [leaf.weight]
        model.leaf_modules.return_value = {"layer": leaf}
        model.trainable_parameters.return_value = {
            "layer.scales": MagicMock(size=3),
            "layer.biases": MagicMock(size=2),
        }
        events = []

        report(model, lambda kind, fields: events.append((kind, fields)))

        self.assertEqual(
            [("metric", {"name": "trainable_parameters", "value": 5})],
            events,
        )


if __name__ == "__main__":
    unittest.main()
