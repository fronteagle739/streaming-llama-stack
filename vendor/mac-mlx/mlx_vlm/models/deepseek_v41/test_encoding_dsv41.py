"""Standalone V4.1 golden-vector, tokenizer, and DSML round-trip checks.

Run with ``python -B test_encoding_dsv41.py``; no MLX or pytest is required.
Set DSV41_ENCODING_TESTS_DIR to the reference encoding/tests directory and
DSV41_TOKENIZER_DIR to a directory containing tokenizer.json (or the file).
Missing external fixtures are reported as skips with their resolved paths.
Compatible with Python 3.12 and later. Import by path avoids mlx_vlm/__init__.
"""

import copy
import difflib
import importlib.util
import json
import os
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("encoding_dsv41.py")
SPEC = importlib.util.spec_from_file_location("_encoding_dsv41", MODULE_PATH)
enc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(enc)

VECTORS_DIR = Path(os.environ.get(
    "DSV41_ENCODING_TESTS_DIR", "./tmp/upstream_dsv41/encoding/tests"
)).resolve()
TOKENIZER_PATH = Path(os.environ.get(
    "DSV41_TOKENIZER_DIR",
    "C:./AppData/Local/Temp/claude/"
    "d--GitHub-RTX-ACCELERATED-MAC-PREFILL-LLAMA/"
    "5008b92d-5137-4669-8b83-1f5d882708cd/scratchpad/dsv41_tok",
))
if TOKENIZER_PATH.suffix != ".json":
    TOKENIZER_PATH /= "tokenizer.json"
TOKENIZER_PATH = TOKENIZER_PATH.resolve()
VECTOR_INPUTS = sorted(VECTORS_DIR.glob("test_input_*.json"))


def assert_exact(expected, actual, label):
    """Quote the first differing character and show an unmodified unified diff."""
    if expected == actual:
        return
    offset = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
                  min(len(expected), len(actual)))
    diff = "".join(difflib.unified_diff(
        expected.splitlines(keepends=True), actual.splitlines(keepends=True),
        fromfile=f"{label}: expected", tofile=f"{label}: actual",
    ))
    raise AssertionError(
        f"{label}: first mismatch at character {offset}\n"
        f"expected: {expected[max(0, offset - 40):offset + 80]!r}\n"
        f"actual:   {actual[max(0, offset - 40):offset + 80]!r}\n{diff}"
    )


def render_vector(input_path):
    cases = enc.load_cases(str(input_path))
    # The upstream format pairs one conversation with each output text file.
    # Fail explicitly if that changes instead of silently testing only case 0.
    assert len(cases) == 1, f"Expected one case in {input_path}, got {len(cases)}"
    return enc.encode_case(cases[0], thinking_mode="chat")[0]


class GoldenVectors(unittest.TestCase):
    pass


class TokenizerRoundTrips(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TOKENIZER_PATH.is_file():
            raise unittest.SkipTest(f"Tokenizer absent: {TOKENIZER_PATH}")
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise unittest.SkipTest(f"tokenizers unavailable; tokenizer: {TOKENIZER_PATH}")
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        model = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))["model"]
        unknown_tokens = {"<unk>", "[UNK]", "<｜unk｜>"}
        if model.get("unk_token"):
            unknown_tokens.add(model["unk_token"])
        cls.unknown_ids = {
            token_id for token in unknown_tokens
            if (token_id := cls.tokenizer.token_to_id(token)) is not None
        }
        if model.get("unk_id") is not None:
            cls.unknown_ids.add(model["unk_id"])


def golden_test(input_path):
    def test(self):
        output_path = input_path.with_name(input_path.stem.replace("input", "output") + ".txt")
        self.assertTrue(output_path.is_file(), f"Missing golden output: {output_path}")
        expected = output_path.read_bytes().decode("utf-8")
        assert_exact(expected, render_vector(input_path), input_path.name)
    return test


def tokenizer_test(input_path):
    def test(self):
        prompt = render_vector(input_path)
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        self.assertTrue(encoded.ids, input_path.name)
        self.assertFalse(self.unknown_ids.intersection(encoded.ids),
                         f"Unknown token in {input_path.name}: {encoded.tokens}")
        decoded = self.tokenizer.decode(encoded.ids, skip_special_tokens=False)
        assert_exact(prompt, decoded, f"{input_path.name}: tokenizer round-trip")
    return test


if VECTOR_INPUTS:
    for input_path in VECTOR_INPUTS:
        name = "test_" + input_path.stem.removeprefix("test_input_")
        setattr(GoldenVectors, name, golden_test(input_path))
        setattr(TokenizerRoundTrips, name, tokenizer_test(input_path))
else:
    def missing_vectors(self):
        self.skipTest(f"Reference vectors absent: {VECTORS_DIR}/test_input_*.json")
    GoldenVectors.test_vectors_absent = missing_vectors
    TokenizerRoundTrips.test_vectors_absent = missing_vectors


class CompletionRoundTrips(unittest.TestCase):
    def test_namespaced_dsml_chat_and_thinking(self):
        for mode in ("chat", "thinking"):
            with self.subTest(mode=mode):
                messages = [
                    {"role": "user", "content": "Look up café"},
                    {"role": "assistant", "content": "Result", "reasoning_content": "Check",
                     "tool_calls": [{"type": "function", "namespace": "search",
                                     "function": {"name": "lookup", "arguments": json.dumps({
                                         "query": "café", "limit": 2, "enabled": True,
                                         "options": {"tags": ["a", "b"]}, "empty": None,
                                     })}}]},
                ]
                original = copy.deepcopy(messages)
                completion = enc.render_message(1, messages, thinking_mode=mode)
                parsed = enc.parse_message_from_completion_text(completion, thinking_mode=mode)
                call = parsed["tool_calls"][0]
                self.assertEqual(call["namespace"], "search")
                self.assertEqual(call["function"]["name"], "lookup")
                self.assertEqual(json.loads(call["function"]["arguments"]),
                                 json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]))
                assert_exact(completion, enc.encode_messages(
                    [parsed], thinking_mode=mode, context=messages[:1]), f"DSML {mode}")
                self.assertEqual(messages, original)


if __name__ == "__main__":
    print(f"Reference vectors discovered: {len(VECTOR_INPUTS)} ({VECTORS_DIR.as_posix()})",
          flush=True)
    unittest.main(verbosity=2)
