import unittest

import mlx.core as mx

import mlx_lm.quant.dwq as dwq


class TestDWQSafety(unittest.TestCase):
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
        self.assertIsNone(
            dwq.emit_dwq_event(None, "validation_start", iteration=0)
        )

    def test_nonfinite_loss_is_rejected_with_iteration(self):
        guard = getattr(dwq, "ensure_finite", None)
        self.assertTrue(callable(guard), "DWQ finite guard is missing")
        with self.assertRaisesRegex(FloatingPointError, "iteration 9.*loss"):
            guard(mx.array(float("nan")), label="loss", iteration=9)


if __name__ == "__main__":
    unittest.main()
