import unittest

import mlx.core as mx

import mlx_lm.quant.dwq as dwq


class TestDWQSafety(unittest.TestCase):
    def test_nonfinite_loss_is_rejected_with_iteration(self):
        guard = getattr(dwq, "ensure_finite", None)
        self.assertTrue(callable(guard), "DWQ finite guard is missing")
        with self.assertRaisesRegex(FloatingPointError, "iteration 9.*loss"):
            guard(mx.array(float("nan")), label="loss", iteration=9)


if __name__ == "__main__":
    unittest.main()
