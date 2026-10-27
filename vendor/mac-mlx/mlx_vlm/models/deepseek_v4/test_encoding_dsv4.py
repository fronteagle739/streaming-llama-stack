"""DeepSeek-V4 prompt-encoding conformance test.

Runs the DeepSeek-V4-Flash-DSpark release's four OFFICIAL encoding test
vectors against the vendored ``encoding_dsv4.py`` reference (copied verbatim
from ``deepseek-ai/DeepSeek-V4-Flash-DSpark/encoding/encoding_dsv4.py``). The
input/gold fixtures under ``encoding_test_vectors/`` are the release's own
``encoding/tests/test_input_*.json`` + ``test_output_*.txt``.

Pure-python (typing/copy/json/re only) — no mlx, no numpy, no weights — so it
runs on the Windows dev box as well as the Mac:

  pytest mlx_vlm/models/deepseek_v4/test_encoding_dsv4.py   (Mac)
  python mlx_vlm/models/deepseek_v4/test_encoding_dsv4.py   (any box)

NOTE: every file is opened with ``encoding="utf-8"`` explicitly. The special
tokens use full-width bars (``｜``); the upstream test relied on a UTF-8 default
locale and therefore fails under Windows' cp1252 default. The assertions and
gold vectors are otherwise identical to the upstream suite.
"""

import json
import os

try:
    from mlx_vlm.models.deepseek_v4.encoding_dsv4 import (
        encode_messages,
        parse_message_from_completion_text,
    )
except ImportError:  # no mlx on this box: import the (mlx-free) module by path
    import importlib.util

    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoding_dsv4.py")
    _spec = importlib.util.spec_from_file_location("_dsv4_encoding", _p)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    encode_messages = _m.encode_messages
    parse_message_from_completion_text = _m.parse_message_from_completion_text

VECTORS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "encoding_test_vectors"
)


def _load_json(name: str):
    with open(os.path.join(VECTORS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _load_text(name: str) -> str:
    with open(os.path.join(VECTORS_DIR, name), encoding="utf-8") as f:
        return f.read()


def test_case_1():
    """Thinking mode with tool calls (multi-turn, tool results merged into user)."""
    td = _load_json("test_input_1.json")
    messages = td["messages"]
    messages[0]["tools"] = td["tools"]
    gold = _load_text("test_output_1.txt")
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert prompt == gold

    marker = "<｜Assistant｜><think>"
    first_start = prompt.find(marker) + len(marker)
    first_end = prompt.find("<｜User｜>", first_start)
    parsed_tc = parse_message_from_completion_text(
        prompt[first_start:first_end], thinking_mode="thinking"
    )
    assert (
        parsed_tc["reasoning_content"]
        == "The user wants to know the weather in Beijing. I should use the get_weather tool."
    )
    assert parsed_tc["content"] == ""
    assert len(parsed_tc["tool_calls"]) == 1
    assert parsed_tc["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(parsed_tc["tool_calls"][0]["function"]["arguments"]) == {
        "location": "Beijing",
        "unit": "celsius",
    }

    last_start = prompt.rfind(marker) + len(marker)
    parsed_final = parse_message_from_completion_text(
        prompt[last_start:], thinking_mode="thinking"
    )
    assert (
        parsed_final["reasoning_content"]
        == "Got the weather data. Let me format a nice response."
    )
    assert "22°C" in parsed_final["content"]
    assert parsed_final["tool_calls"] == []


def test_case_2():
    """Thinking mode without tools (drop_thinking removes earlier reasoning)."""
    messages = _load_json("test_input_2.json")
    gold = _load_text("test_output_2.txt")
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert prompt == gold

    marker = "<｜Assistant｜><think>"
    last_start = prompt.rfind(marker) + len(marker)
    parsed = parse_message_from_completion_text(
        prompt[last_start:], thinking_mode="thinking"
    )
    assert (
        parsed["reasoning_content"]
        == "The user asks about the capital of France. It is Paris."
    )
    assert parsed["content"] == "The capital of France is Paris."
    assert parsed["tool_calls"] == []
    assert "The user said hello" not in prompt


def test_case_3():
    """Interleaved thinking + search (developer with tools, latest_reminder)."""
    messages = _load_json("test_input_3.json")
    gold = _load_text("test_output_3.txt")
    assert encode_messages(messages, thinking_mode="thinking") == gold


def test_case_4():
    """Quick instruction task with latest_reminder (chat mode, action task)."""
    messages = _load_json("test_input_4.json")
    gold = _load_text("test_output_4.txt")
    assert encode_messages(messages, thinking_mode="chat") == gold


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ENCODING_DSV4_TEST ALL PASS ({len(fns)} tests)")


if __name__ == "__main__":
    main()
