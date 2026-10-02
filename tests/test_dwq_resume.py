# Copyright © 2026 Andrew M. Cox.

import unittest
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm.quant.dwq import dwq_quantize


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(32, 8, bias=False).to_quantized(
            group_size=32, bits=4
        )
        self.freeze()

    def __call__(self, tokens):
        return self.projection(mx.eye(32)[tokens])


class TestDWQResume(unittest.TestCase):
    def test_interrupt_before_first_update_does_not_save_completed_checkpoint(self):
        checkpoint = Mock()

        def target(tokens, index, split, lengths=None):
            if split == "train":
                raise KeyboardInterrupt
            return mx.zeros((*tokens.shape, 8))

        with self.assertRaises(KeyboardInterrupt):
            dwq_quantize(
                TinyStudent(),
                target,
                optim.Adam(learning_rate=0.001),
                train_data=[([1, 2, 3, 4], 0)],
                valid_data=[([1, 2, 3, 4], 0)],
                batch_size=1,
                max_seq_length=8,
                seed=7,
                dtype=mx.float32,
                checkpoint_interval=1,
                checkpoint_fn=checkpoint,
            )
        checkpoint.assert_not_called()

    def test_nonfinite_reduced_loss_aborts_before_completed_checkpoint(self):
        phase = ["valid"]
        events = []
        checkpoint = Mock()
        teacher = mx.sin(mx.arange(32 * 8).reshape(32, 8).astype(mx.float32))

        def target(tokens, index, split, lengths=None):
            phase[0] = split
            return mx.eye(32)[tokens] @ teacher

        def reduce(value, **kwargs):
            if (
                phase[0] == "train"
                and value.ndim == 0
                and mx.issubdtype(value.dtype, mx.floating)
            ):
                self.assertTrue(mx.isfinite(value).item())
                return mx.array(float("nan"))
            return value

        with patch("mlx_lm.quant.dwq.mx.distributed.all_sum", side_effect=reduce):
            with self.assertRaises(FloatingPointError):
                dwq_quantize(
                    TinyStudent(),
                    target,
                    optim.Adam(learning_rate=0.001),
                    train_data=[([1, 2, 3, 4], 0)],
                    valid_data=[([1, 2, 3, 4], 0)],
                    batch_size=1,
                    max_seq_length=8,
                    seed=7,
                    dtype=mx.float32,
                    checkpoint_interval=1,
                    checkpoint_fn=checkpoint,
                    event_fn=lambda kind, fields: events.append((kind, fields)),
                )
        checkpoint.assert_not_called()
        self.assertTrue(any(kind == "nonfinite_loss" for kind, _ in events))
        self.assertFalse(any(kind == "training_iteration" for kind, _ in events))

    def test_interruption_then_resume_matches_uninterrupted_adam_trajectory(self):
        dataset = [
            ([1, 2, 3, 4], 0),
            ([2, 3, 4, 5], 0),
            ([3, 4, 5, 6], 0),
            ([4, 5, 6, 7], 0),
        ]
        teacher = mx.sin(mx.arange(32 * 8).reshape(32, 8).astype(mx.float32))

        def target(tokens, index, split, lengths=None):
            return mx.eye(32)[tokens] @ teacher

        def model():
            mx.random.seed(42)
            return TinyStudent()

        common = dict(
            train_data=dataset,
            valid_data=dataset[:1],
            batch_size=1,
            max_seq_length=8,
            seed=7,
            dtype=mx.float32,
            checkpoint_interval=1,
            validation_interval=1,
        )
        reference = model()
        reference_optimizer = optim.Adam(learning_rate=0.001, bias_correction=True)
        dwq_quantize(reference, target, reference_optimizer, **common)

        snapshots = []

        def save(completed, params, state, initial_loss, valid_loss):
            mx.eval(params, state)
            snapshots.append(
                (
                    completed,
                    tree_map(lambda value: value, params),
                    tree_map(lambda value: value, state),
                    initial_loss,
                    valid_loss,
                )
            )

        def interrupt(tokens, index, split, lengths=None):
            if split == "train" and index == 2:
                raise KeyboardInterrupt
            return target(tokens, index, split, lengths)

        interrupted = model()
        with self.assertRaises(KeyboardInterrupt):
            dwq_quantize(
                interrupted,
                interrupt,
                optim.Adam(learning_rate=0.001, bias_correction=True),
                checkpoint_fn=save,
                **common,
            )
        completed, params, state, initial_loss, valid_loss = snapshots[-1]
        self.assertEqual(completed, 2)

        resumed = model()
        resumed_optimizer = optim.Adam(learning_rate=0.001, bias_correction=True)
        dwq_quantize(
            resumed,
            target,
            resumed_optimizer,
            start_iteration=completed,
            initial_params=params,
            optimizer_state=state,
            initial_valid_loss=initial_loss,
            **common,
        )
        for expected, actual in (
            (reference.parameters(), resumed.parameters()),
            (reference_optimizer.state, resumed_optimizer.state),
        ):
            first = tree_flatten(expected)
            second = tree_flatten(actual)
            self.assertEqual([key for key, _ in first], [key for key, _ in second])
            for (key, value), (_, other) in zip(first, second):
                if isinstance(value, mx.array):
                    self.assertTrue(
                        mx.allclose(value, other, rtol=1e-6, atol=1e-6).item(), key
                    )
                else:
                    self.assertEqual(value, other)


if __name__ == "__main__":
    unittest.main()
