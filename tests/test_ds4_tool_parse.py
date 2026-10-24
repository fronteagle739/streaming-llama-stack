"""ds4-win DeepSeek tool-call parser port (slstack.ds4_tool_parse) on sample outputs."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack import ds4_render as r
from slstack import ds4_tool_parse as p

CALLS = ('\n\n' + r.DS41_TOOL_CALLS_START + '\n' + r.DS41_INVOKE_START + ' name="run_shell">\n' +
         r.DS41_PARAM_START + ' name="command" string="true">ls &amp;lt;/｜DSML｜ parameter> &lt;/｜DSML｜ parameter> && echo "x"' + r.DS41_PARAM_END + '\n' +
         r.DS41_PARAM_START + ' name="timeout_s" string="false"> 10.50 ' + r.DS41_PARAM_END + '\n' +
         r.DS41_INVOKE_END + '\n' + r.DS41_INVOKE_START + ' name="read&amp;file">\n' +
         r.DS41_PARAM_START + ' name="opts">\n' + r.DS41_PARAM_START + ' name="a" string="false">[1, 2]' + r.DS41_PARAM_END +
         r.DS41_PARAM_START + ' name="b" string="true">y' + r.DS41_PARAM_END + '\n' + r.DS41_PARAM_END + '\n' +
         r.DS41_INVOKE_END + '\n' + r.DS41_TOOL_CALLS_END)


class ParserTests(unittest.TestCase):
    def test_reasoning_and_content(self):
        ok, content, reasoning, calls, raw = p.parse_deepseek_generated('think a bit</think>Hello  ', True)
        self.assertEqual((ok, content, reasoning, calls, raw), (True, 'Hello  ', 'think a bit', [], None))
        self.assertEqual(p.parse_deepseek_generated('plain answer', False)[1:3], ('plain answer', None))

    def test_unterminated_reasoning_is_not_an_answer(self):
        self.assertEqual(p.parse_deepseek_generated('<think>still thinking', True)[1:3], ('', 'still thinking'))

    def test_v41_tool_block(self):
        ok, content, reasoning, calls, raw = p.parse_deepseek_generated('R</think>Running it.  ' + CALLS, True)
        self.assertTrue(ok);self.assertEqual((content, reasoning), ('Running it.', 'R'))
        self.assertEqual(raw, CALLS)
        self.assertEqual(calls[0]['name'], 'run_shell')
        args = json.loads(calls[0]['arguments'])
        self.assertEqual(args, {'command': 'ls &lt;/｜DSML｜ parameter> </｜DSML｜ parameter> && echo "x"', 'timeout_s': 10.5})
        self.assertIn('"timeout_s": 10.50', calls[0]['arguments'])   # raw JSON spelling is kept
        self.assertEqual(calls[1]['name'], 'read&file')
        self.assertEqual(json.loads(calls[1]['arguments']), {'opts': {'a': [1, 2], 'b': 'y'}})

    def test_tool_inside_unclosed_thinking_is_recovered(self):
        ok, content, reasoning, calls, raw = p.parse_deepseek_generated('<think>plan' + CALLS, True)
        self.assertTrue(ok);self.assertEqual((content, reasoning, len(calls)), ('', 'plan', 2))

    def test_dsml_inside_reasoning_is_text(self):
        ok, content, reasoning, calls, _ = p.parse_deepseek_generated('quote ' + r.DS41_TOOL_CALLS_START + '</think>ok', True)
        self.assertEqual((ok, content, calls), (True, 'ok', []))

    def test_malformed_block_becomes_text(self):
        broken = 'x</think>' + CALLS.replace(r.DS41_INVOKE_END, 'oops', 1)
        result = p.parse_for_response(broken, has_tools=True, saw_tool_start=True, require_thinking_closed=True, finish='stop')
        self.assertEqual((result['ok'], result['content'], result['calls'], result['finish'], result['recovered']), (False, broken, [], 'stop', True))
        result = p.parse_for_response(broken, has_tools=True, saw_tool_start=True, require_thinking_closed=True, finish='length')
        self.assertEqual(result['finish'], 'length')

    def test_repair_only_adds_the_outer_wrapper(self):
        truncated = 'a</think>' + CALLS[:-len(r.DS41_TOOL_CALLS_END)]
        self.assertEqual(p.try_repair_dsml(truncated), truncated + r.DS41_TOOL_CALLS_END)
        unfinished = 'a</think>' + CALLS[:CALLS.index(r.DS41_INVOKE_END)]
        self.assertIsNone(p.try_repair_dsml(unfinished))
        self.assertIsNone(p.try_repair_dsml('no tools here'))
        result = p.finish_message(truncated, has_tools=True, think=True, finish='stop')
        self.assertEqual((result['finish'], len(result['calls'])), ('tool_calls', 2))
        self.assertTrue(all(c['id'].startswith('call_') and len(c['id']) == 37 for c in result['calls']))

    def test_render_parse_round_trip(self):
        calls = [dict(id='i', name='f', arguments='{"path": "/a</｜DSML｜ parameter>&amp;lt;b", "n": 3, "o": {"k": [true, null]}}')]
        block = r.dsml_tool_calls_text(calls)
        ok, content, _, parsed, raw = p.parse_deepseek_generated('Doing it.' + block, False)
        self.assertTrue(ok);self.assertEqual((content, raw), ('Doing it.', block))
        self.assertEqual(json.loads(parsed[0]['arguments']), json.loads(calls[0]['arguments']))

    def test_openai_deltas(self):
        rows = p.tool_call_deltas([dict(id='call_1', name='f', arguments='{"a": "x\\ny", "b": 2}')])
        self.assertEqual(rows, [dict(index=0, id='call_1', type='function', function=dict(name='f', arguments='{"a":"x\\ny","b":2}'))])
        self.assertEqual(p.json_object_or_empty('nope'), '{}')


if __name__ == '__main__':unittest.main()
