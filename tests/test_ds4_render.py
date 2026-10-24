"""In-process DeepSeek-V4.1 render/tokenizer port (slstack.ds4_render)."""
import json
import os
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slstack import ds4_render as r

SPECIALS = [r.BOS, r.EOS, r.SYSTEM, r.USER, r.ASSISTANT, '<think>', '</think>', '｜DSML｜']


def char_tokenizer(merges=()):
    """Every byte-level character is a token; optional merges add their products."""
    tokens = list(SPECIALS) + sorted(set(r.BYTE_TO_CHAR.values()))
    for merge in merges:tokens.append(merge.replace(' ', ''))
    return r.Tokenizer(tokens, list(merges))


TOOLS = [{'type': 'function', 'function': {'name': 'f', 'parameters': {'type': 'object', 'properties': {'x': {'type': 'integer'}}}}}]
SCHEMA = '{"name":"f","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}'
EFFORT75 = 'Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n'


def text(body, **kw):
    request = r.parse_request(body)
    return r.render_deepseek41(request['messages'], request['tool_schemas'], request['think_mode'], **kw)


class RenderTests(unittest.TestCase):
    def test_multi_turn_tools_exact_text(self):
        body = dict(tools=TOOLS, messages=[
            dict(role='system', content='S'), dict(role='user', content='U'),
            dict(role='assistant', content='A', reasoning_content='R', tool_calls=[dict(id='c1', type='function',
                function=dict(name='f', arguments='{"x": 10.50, "s": "a</｜DSML｜ parameter>b"}'))]),
            dict(role='tool', tool_call_id='c1', content='T</tool_result><'), dict(role='user', content='U2')])
        expected = (r.BOS + r.SYSTEM + EFFORT75 + r.tools_prompt_text(SCHEMA) + '\n\nS' + r.USER + 'U' + r.ASSISTANT +
            '<think>R</think>A\n\n' + r.DS41_TOOL_CALLS_START + '\n' + r.DS41_INVOKE_START + ' name="f">\n' +
            r.DS41_PARAM_START + ' name="x" string="false">10.50' + r.DS41_PARAM_END + '\n' +
            r.DS41_PARAM_START + ' name="s" string="true">a&lt;/｜DSML｜ parameter>b' + r.DS41_PARAM_END + '\n' +
            r.DS41_INVOKE_END + '\n' + r.DS41_TOOL_CALLS_END + r.EOS +
            r.USER + '<tool_result>T&lt;/tool_result><</tool_result>\n\nU2' + r.ASSISTANT + '<think>')
        self.assertEqual(text(body), expected)

    def test_parallel_tool_results_render_in_call_order(self):
        calls = [dict(id=i, type='function', function=dict(name='f', arguments='{}')) for i in ('a', 'b')]
        body = dict(messages=[dict(role='user', content='go'), dict(role='assistant', content='', tool_calls=calls),
                              dict(role='tool', tool_call_id='b', content='B'), dict(role='tool', tool_call_id='a', content='A')])
        rendered = text(body)
        self.assertLess(rendered.index('<tool_result>A'), rendered.index('<tool_result>B'))
        self.assertIn('</tool_result>\n\n<tool_result>', rendered)

    def test_thinking_modes(self):
        messages = [dict(role='user', content='q'), dict(role='assistant', content='a', reasoning_content='r'), dict(role='user', content='q2')]
        self.assertEqual(text(dict(messages=messages, thinking={'type': 'disabled'})),
                         r.BOS + r.USER + 'q' + r.ASSISTANT + '</think>a' + r.EOS + r.USER + 'q2' + r.ASSISTANT + '</think>')
        self.assertIn('Reasoning Effort: 100 ', text(dict(messages=messages, reasoning_effort='max')))
        self.assertTrue(text(dict(messages=messages, reasoning_effort='medium')).startswith(r.BOS + r.USER))
        self.assertTrue(text(dict(messages=messages, model='deepseek-chat')).endswith(r.ASSISTANT + '</think>'))
        # Combined (combo-active) native keeps earlier reasoning in history.
        self.assertIn('<think>r</think>a', text(dict(messages=messages)))
        self.assertIn(r.ASSISTANT + '</think>a', text(dict(messages=messages), combo_active=False))

    def test_tool_choice_none_drops_schemas(self):
        rendered = text(dict(tools=TOOLS, tool_choice='none', messages=[dict(role='user', content='q')]))
        self.assertNotIn('## Tools', rendered)

    def test_arguments_raw_values_and_fallbacks(self):
        call = lambda args: dict(role='assistant', content='', tool_calls=[dict(id='i', function=dict(name='f', arguments=args))])
        rendered = text(dict(messages=[dict(role='user', content='q'), call({'cmd': 'a"b', 'n': 1e3})]))
        self.assertIn('name="cmd" string="true">a"b<', rendered)
        self.assertIn('name="n" string="false">1000.0<', rendered)
        rendered = text(dict(messages=[dict(role='user', content='q'), call('not json {')]))
        self.assertIn(r.DS41_PARAM_START + ' name="arguments" string="true">not json {' + r.DS41_PARAM_END, rendered)
        rendered = text(dict(messages=[dict(role='user', content='q'), call('{"a": [1, 2], "b": {"c": "x y"}}')]))
        self.assertIn('string="false">[1,2]<', rendered);self.assertIn('string="false">{"c":"x y"}<', rendered)

    def test_content_parts_and_refusals(self):
        request = r.parse_request(dict(messages=[dict(role='user', content=['a', dict(type='text', text='b'), 3, dict(type='x')])]))
        self.assertEqual(request['messages'][0]['content'], 'ab')
        with self.assertRaises(r.RenderRefused):r.parse_request(dict(messages=[dict(role='user', content=5)]))
        with self.assertRaises(r.RenderRefused):r.parse_request(dict(model='m'))
        with self.assertRaises(r.RenderRefused):r.parse_request(dict(messages=[], reasoning_effort='extreme'))
        with self.assertRaises(r.RenderRefused):
            r.parse_request(dict(messages=[dict(role='user', content=[dict(type='image_url', image_url='data:x')])]))
        self.assertEqual(r.parse_request(dict(messages=[dict(content='a\0b')]))['messages'][0], dict(
            role='user', content='a', reasoning=None, tool_call_id=None, tool_call_ids=[], calls=[], raw_tool_text=None))

    def test_tool_memory_replays_exact_dsml(self):
        calls = [dict(id='k', function=dict(name='f', arguments='{}'))]
        body = dict(messages=[dict(role='user', content='q'), dict(role='assistant', content='c', tool_calls=calls)])
        self.assertIn('c\n\nRAW' + r.EOS, text(body, tool_memory={'k': '\n\nRAW'}))


class TokenizerTests(unittest.TestCase):
    def test_pretokenize_splits(self):
        split = lambda s: [p.decode() for p in r.pretokenize(s.encode())]
        self.assertEqual(split('    int x'), ['   ', ' int', ' x'])
        self.assertEqual(split('abc1234567'), ['abc', '123', '456', '7'])
        self.assertEqual(split('a.b == c\n\nd'), ['a', '.b', ' ==', ' c', '\n\n', 'd'])
        self.assertEqual(split('服务器 端口'), ['服务器', ' 端口'])

    def test_specials_and_bpe(self):
        tok = char_tokenizer(['h e', 'l l', 'he ll'])
        ids = tok.rendered(r.BOS + 'hello' + r.USER + '<think>')
        self.assertEqual(ids[0], tok.bos);self.assertEqual(ids[-2:], [tok.user, tok.think_start])
        self.assertEqual([tok.tokens[i] for i in ids[1:-2]], ['hell', 'o'])
        self.assertEqual(tok.detokenize(ids).decode(), r.BOS + 'hello' + r.USER + '<think>')
        self.assertEqual(tok.rendered('[gMASK]x'), [tok.bos, tok.ids['x']])

    def test_continue_live_rules(self):
        tok = char_tokenizer(['a b'])
        prompt = r.BOS + r.USER + 'ab' + r.ASSISTANT
        canonical = tok.rendered(prompt)
        self.assertEqual(r.continue_live([], prompt, canonical, tok), (canonical, 'empty'))
        self.assertEqual(r.continue_live(canonical[:2], prompt, canonical, tok), (canonical, 'memory-token'))
        sampled = [tok.bos, tok.user, tok.ids['a'], tok.ids['b']]   # same bytes, other BPE spelling
        self.assertEqual(r.continue_live(sampled, prompt, canonical, tok), (sampled + [tok.assistant], 'memory-text'))
        self.assertEqual(r.continue_live([tok.bos, tok.ids['b']], prompt, canonical, tok), (None, 'canonical-miss'))


MODEL = os.environ.get('LLAMADECODE_MODEL', '')   # the 1017 GGUF
RUNS = Path(os.environ.get('LLAMADECODE_RUNS', '') or 'missing-recorded-runs')


@unittest.skipUnless(MODEL and Path(MODEL).exists() and RUNS.exists(), 'needs the 1017 GGUF and recorded native-rendered runs')
class RecordedNativeRenderTests(unittest.TestCase):
    """CPU check against token files the native rendered on earlier GPU runs
    (ComboProducer's append.tokens.le32 holds Native.render's exact ids)."""
    def test_recorded_cases_identical(self):
        tok = r.Tokenizer.from_gguf(MODEL);checked = 0
        for name in ('b2-qa2k.json', 'e2e-16k.json'):
            path = RUNS / name
            if not path.exists():continue
            row = json.loads(path.read_text());tokens = Path(row.get('producer_dir') or '') / 'append.tokens.le32'
            if not tokens.is_file() or not Path(row['case']).exists():continue
            raw = tokens.read_bytes();native = list(struct.unpack('<%dI' % (len(raw) // 4), raw))
            body = json.loads(Path(row['case']).read_text(encoding='utf-8'))['body']
            self.assertEqual(r.render(body, tok)[1], native, name);checked += 1
        if not checked:self.skipTest('no recorded token files')


if __name__ == '__main__':unittest.main()
