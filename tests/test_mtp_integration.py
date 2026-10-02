# Copyright © 2026 Andrew M. Cox.

import contextlib
import importlib
import io
import itertools
import sys
import unittest
import warnings
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.generate import (
    BatchGenerator,
    generate,
    generate_step,
    mtp_generate_step,
    stream_generate,
)
from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import does_model_support_mtp, quantize_model
from test_mtp import _make_qwen3_5_mtp_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def tokenizer():
    backend = Tokenizer(
        models.WordLevel({f"t{i}": i for i in range(256)}, unk_token="t0")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return TokenizerWrapper(
        PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t0")
    )


class TestMTPIntegration(unittest.TestCase):
    def setUp(self):
        mx.random.seed(7)
        self.model = _make_qwen3_5_mtp_model()

    def test_chunked_prefill_preserves_processor_history_and_greedy_output(self):
        prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
        histories = []

        def processor(tokens, logits):
            histories.append(tokens.tolist())
            token = sum(tokens.tolist()) % 256
            return logits + (mx.arange(256) == token) * 1000

        expected = [
            token
            for token, _ in generate_step(
                prompt,
                self.model,
                max_tokens=8,
                logits_processors=[processor],
                prefill_step_size=2,
            )
        ]
        histories.clear()
        actual = [
            token
            for token, _, _ in mtp_generate_step(
                prompt,
                self.model,
                max_tokens=8,
                logits_processors=[processor],
                prefill_step_size=2,
            )
        ]
        self.assertEqual(histories[0], prompt.tolist())
        self.assertEqual(actual, expected)

    def test_final_input_embedding_is_used_with_and_without_chunked_prefill(self):
        for ids in ([3], [1, 2, 3, 4]):
            with self.subTest(ids=ids):
                prompt = mx.array(ids, dtype=mx.uint32)
                embeddings = self.model.language_model.model.embed_tokens(prompt)
                embeddings = embeddings.at[-1].multiply(0)
                expected = [
                    token
                    for token, _ in generate_step(
                        prompt,
                        self.model,
                        max_tokens=6,
                        input_embeddings=embeddings,
                        prefill_step_size=2,
                    )
                ]
                actual = [
                    token
                    for token, _, _ in mtp_generate_step(
                        prompt,
                        self.model,
                        max_tokens=6,
                        input_embeddings=embeddings,
                        prefill_step_size=2,
                    )
                ]
                self.assertEqual(actual, expected)
                if len(ids) == 1:
                    ordinary = [
                        token
                        for token, _ in generate_step(prompt, self.model, max_tokens=6)
                    ]
                    self.assertNotEqual(expected, ordinary)

    def test_embedding_only_mtp_rejects_before_generation(self):
        with self.assertRaisesRegex(ValueError, "prompt token IDs"):
            list(
                mtp_generate_step(
                    mx.array([], dtype=mx.uint32),
                    self.model,
                    input_embeddings=mx.zeros((2, 64)),
                )
            )

    def test_accepted_drafts_preserve_backbone_cache_state(self):
        prompt = mx.array([1, 2, 3, 4], dtype=mx.uint32)
        normal_cache = make_prompt_cache(self.model)
        combined_cache = make_prompt_cache(self.model) + self.model.make_mtp_cache()

        def force_zero(_, logits):
            return mx.zeros_like(logits).at[..., 0].add(1000)

        before = getattr(self.model, "mtp_total_accepted", 0)
        expected = [
            token
            for token, _ in generate_step(
                prompt,
                self.model,
                max_tokens=8,
                prompt_cache=normal_cache,
                logits_processors=[force_zero],
            )
        ]
        actual = [
            token
            for token, _, _ in mtp_generate_step(
                prompt,
                self.model,
                max_tokens=8,
                prompt_cache=combined_cache,
                logits_processors=[force_zero],
            )
        ]
        self.assertEqual(actual, expected)
        self.assertGreater(self.model.mtp_total_accepted, before)
        for ordinary, speculative in zip(normal_cache, combined_cache):
            for (key, first), (other_key, second) in zip(
                tree_flatten(ordinary.state), tree_flatten(speculative.state)
            ):
                self.assertEqual(key, other_key)
                if isinstance(first, mx.array):
                    self.assertTrue(mx.allclose(first, second, atol=1e-5).item())
                else:
                    self.assertEqual(first, second)

    def test_quantization_replacements_remain_in_retained_cache(self):
        cache = make_prompt_cache(self.model)
        main_count = len(cache)
        original = list(cache)
        list(
            mtp_generate_step(
                mx.array([1, 2, 3], mx.uint32),
                self.model,
                prompt_cache=cache,
                max_tokens=4,
                kv_bits=4,
                kv_group_size=32,
                quantized_kv_start=0,
            )
        )
        for first, current in zip(original, cache[:main_count]):
            if hasattr(first, "to_quantized"):
                self.assertIsInstance(current, QuantizedKVCache)
                self.assertIsNot(current, first)
        self.assertTrue(
            all(isinstance(entry, QuantizedKVCache) for entry in cache[main_count:])
        )
        list(
            mtp_generate_step(
                mx.array([4], mx.uint32),
                self.model,
                prompt_cache=cache,
                max_tokens=2,
                kv_bits=4,
                kv_group_size=32,
                quantized_kv_start=0,
            )
        )
        self.assertEqual(len(cache), main_count + len(self.model.make_mtp_cache()))

    def test_batch_decoding_accepts_mixed_mtp_and_backbone_caches(self):
        cache = make_prompt_cache(self.model)
        cache.extend(self.model.make_mtp_cache())
        ordinary = make_prompt_cache(self.model)
        with contextlib.closing(
            BatchGenerator(self.model, max_tokens=2, prefill_batch_size=2)
        ) as generator:
            uids = generator.insert([[1, 2, 3], [4, 5, 6]], caches=[cache, ordinary])
            responses = []
            while batch := generator.next_generated():
                responses.extend(batch)
        for uid in uids:
            self.assertEqual(
                len([response for response in responses if response.uid == uid]), 2
            )
        self.assertGreater(len(cache), len(ordinary))

    def test_negative_token_limit_matches_ordinary_decode(self):
        prompt = mx.array([1, 2, 3], mx.uint32)
        expected = [
            token
            for token, _ in itertools.islice(
                generate_step(prompt, self.model, max_tokens=-1), 10
            )
        ]
        actual = [
            token
            for token, _, _ in itertools.islice(
                mtp_generate_step(prompt, self.model, max_tokens=-1), 10
            )
        ]
        self.assertEqual(actual, expected)

    def test_prompt_progress_reaches_completion_before_first_token(self):
        progress = []
        generator = stream_generate(
            self.model,
            tokenizer(),
            mx.array([1, 2, 3, 4, 5], mx.uint32),
            mtp=True,
            max_tokens=2,
            prefill_step_size=2,
            prompt_progress_callback=lambda done, total: progress.append((done, total)),
        )
        next(generator)
        self.assertEqual(progress[0], (0, 5))
        self.assertEqual(progress[-1], (5, 5))
        self.assertTrue(all(a[0] <= b[0] for a, b in zip(progress, progress[1:])))
        self.assertGreater(len(progress), 2)
        generator.close()

    def test_backbone_only_cache_retains_and_reuses_mtp_state(self):
        prompt = mx.array([1, 2, 3], mx.uint32)
        cache = make_prompt_cache(self.model)
        main_count = len(cache)
        list(mtp_generate_step(prompt, self.model, prompt_cache=cache, max_tokens=4))
        draft_cache = cache[main_count:]
        self.assertTrue(draft_cache)
        initial_offsets = [entry.offset for entry in draft_cache]
        list(
            mtp_generate_step(
                mx.array([4, 5], mx.uint32),
                self.model,
                prompt_cache=cache,
                max_tokens=2,
            )
        )
        self.assertEqual(len(cache), main_count + len(draft_cache))
        self.assertTrue(
            all(
                first is second
                for first, second in zip(draft_cache, cache[main_count:])
            )
        )
        self.assertTrue(
            all(
                entry.offset > offset
                for entry, offset in zip(draft_cache, initial_offsets)
            )
        )

    def test_stream_dispatch_uses_explicit_stream_and_actual_head(self):
        tokens = mx.array([1, 2, 3], dtype=mx.uint32)
        stream = mx.new_stream(mx.default_device())
        expected = [
            token
            for token, _ in generate_step(tokens, self.model, stream, max_tokens=6)
        ]
        actual = [
            response.token
            for response in stream_generate(
                self.model, tokenizer(), tokens, mtp=True, stream=stream, max_tokens=6
            )
        ]
        self.assertEqual(actual, expected)
        disabled = _make_qwen3_5_mtp_model(mtp_layers=0)
        self.assertFalse(does_model_support_mtp(disabled))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            responses = list(
                stream_generate(
                    disabled, tokenizer(), tokens, mtp=True, stream=stream, max_tokens=6
                )
            )
        self.assertTrue(
            any(issubclass(warning.category, UserWarning) for warning in caught)
        )
        self.assertTrue(all(not response.from_draft for response in responses))

    def test_cli_mtp_sampling_matches_public_generation_arguments(self):
        module = importlib.import_module("mlx_lm.generate")
        wrapped = tokenizer()
        mx.random.seed(42)
        expected = generate(
            self.model,
            wrapped,
            [1, 2],
            mtp=True,
            temp=0.7,
            top_k=3,
            max_tokens=8,
            verbose=False,
        )
        output = io.StringIO()
        argv = [
            "mlx_lm.generate",
            "--model",
            "fixture",
            "--prompt",
            "t1 t2",
            "--mtp",
            "--temp",
            "0.7",
            "--top-k",
            "3",
            "--max-tokens",
            "8",
            "--seed",
            "42",
            "--verbose",
            "false",
        ]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(
                patch.object(module, "load", return_value=(self.model, wrapped))
            )
            stack.enter_context(contextlib.redirect_stdout(output))
            module.main()
        self.assertEqual(output.getvalue().strip(), expected.strip())

    def test_converted_moe_mtp_reload_preserves_all_weights(self):
        model = _make_qwen3_5_mtp_model(model_type="qwen3_5_moe")
        weights = dict(tree_flatten(model.parameters()))
        actual = model.sanitize(dict(weights))
        self.assertEqual(set(actual), set(weights))
        for key in weights:
            self.assertTrue(mx.array_equal(actual[key], weights[key]).item(), key)

    def test_partial_quantization_override_inherits_independent_defaults(self):
        model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))
        model, config = quantize_model(
            model,
            {},
            group_size=32,
            bits=4,
            mode="mxfp4",
            quant_predicate=lambda path, module: {"bits": 4},
        )
        for layer in model.layers:
            self.assertEqual(layer.mode, "mxfp4")
            self.assertEqual(layer.group_size, 32)
        for value in config["quantization"].values():
            if isinstance(value, dict):
                self.assertEqual(value, {"group_size": 32, "bits": 4, "mode": "mxfp4"})


if __name__ == "__main__":
    unittest.main()
