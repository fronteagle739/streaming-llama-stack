"""DeepSeek generated-message parser: a Python port of ds4-win's exact logic.

The parser is C, so this is a line-for-line port, not the process.
Provenance: engines/ds4-win/ds4_server.c at stack revision b5907f2 (fork base
058ae09159a1, merge e1e45374ddc3): find_any_tool_start/end, trim_tool_separator_ws,
find_tool_structural_text, dsml_unescape_text, dsml_attr, tool_call_json_args_add,
dsml_parse_leaf_param_json, dsml_parse_nested_params_object, split_reasoning_content,
ds4_local_unterminated_reasoning, ds4_unterminated_reasoning_before_tool,
parse_deepseek_generated_message_ex, try_repair_dsml,
parse_generated_message_for_response_for_syntax (DeepSeek-V4.1 syntax),
append_json_object_or_empty / append_tool_call_deltas_json, random_tool_id;
ds4_tool_text.h ds4_tool_text_unescape.
"""
from __future__ import annotations
import secrets
from .ds4_render import (DS41_TOOL_CALLS_START, DS41_TOOL_CALLS_END, DS41_INVOKE_START, DS41_INVOKE_END,
                         DS41_PARAM_START, DS41_PARAM_END, json_escape, json_minify_raw_value, json_args_parse, skip_value)

DSML = '｜DSML｜';DSML_SHORT = 'DSML｜'
TOOL_CALLS_START = '<' + DSML + 'tool_calls>';TOOL_CALLS_END = '</' + DSML + 'tool_calls>'
INVOKE_START = '<' + DSML + 'invoke';INVOKE_END = '</' + DSML + 'invoke>'
PARAM_START = '<' + DSML + 'parameter';PARAM_END = '</' + DSML + 'parameter>'
TOOL_CALLS_START_SHORT = '<' + DSML_SHORT + 'tool_calls>';TOOL_CALLS_END_SHORT = '</' + DSML_SHORT + 'tool_calls>'
INVOKE_START_SHORT = '<' + DSML_SHORT + 'invoke';INVOKE_END_SHORT = '</' + DSML_SHORT + 'invoke>'
PARAM_START_SHORT = '<' + DSML_SHORT + 'parameter';PARAM_END_SHORT = '</' + DSML_SHORT + 'parameter>'
SPACE = ' \t\n\r\v\f'
STYLES = {0: (TOOL_CALLS_START, TOOL_CALLS_END, INVOKE_START, INVOKE_END, PARAM_START, PARAM_END),
          1: ('<tool_calls>', '</tool_calls>', '<invoke', '</invoke>', '<parameter', '</parameter>'),
          2: (TOOL_CALLS_START_SHORT, TOOL_CALLS_END_SHORT, INVOKE_START_SHORT, INVOKE_END_SHORT, PARAM_START_SHORT, PARAM_END_SHORT),
          3: (DS41_TOOL_CALLS_START, DS41_TOOL_CALLS_END, DS41_INVOKE_START, DS41_INVOKE_END, DS41_PARAM_START, DS41_PARAM_END)}


def _find(s, needle, start=0):
    i = s.find(needle, start)
    return None if i < 0 else i


def find_any_tool_start(s, start=0):
    hits = [_find(s, n, start) for n in (DS41_TOOL_CALLS_START, TOOL_CALLS_START, TOOL_CALLS_START_SHORT, '<tool_calls>', '<tool_call>')]
    hits = [h for h in hits if h is not None]
    return min(hits) if hits else None


def find_any_tool_end(s, start=0):
    hits = [_find(s, n, start) for n in (DS41_TOOL_CALLS_END, TOOL_CALLS_END, TOOL_CALLS_END_SHORT, '</tool_calls>', '</tool_call>')]
    hits = [h for h in hits if h is not None]
    return min(hits) if hits else None


def skip_ws(s, i):
    while i < len(s) and s[i] in SPACE:i += 1
    return i


WRAPPERS = ((DS41_PARAM_START, DS41_PARAM_END), (PARAM_START, PARAM_END), (PARAM_START_SHORT, PARAM_END_SHORT),
            ('<parameter', '</parameter>'), ('<arg_key>', '</arg_key>'), ('<arg_value>', '</arg_value>'))


def find_tool_structural_text(s, needle, last):
    found = None;i = 0
    while i < len(s):
        if s.startswith(needle, i):
            found = i
            if not last:return found
        skipped = False
        for start, end in WRAPPERS:
            if not s.startswith(start, i):continue
            tag_end = _find(s, '>', i)
            close = _find(s, end, tag_end + 1) if tag_end is not None else None
            if close is None:return found
            i = close + len(end);skipped = True
            break
        if not skipped:i += 1
    return found


def dsml_unescape_text(s):
    out = [];i = 0
    while i < len(s):
        if s[i] != '&':out.append(s[i]);i += 1;continue
        for entity, ch in (('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"'), ('&apos;', "'")):
            if s.startswith(entity, i):out.append(ch);i += len(entity);break
        else:out.append('&');i += 1
    return ''.join(out)


def dsml_attr(tag, name):
    pattern = name + '="';p = tag.find(pattern)
    if p < 0:return None
    p += len(pattern);q = tag.find('"', p)
    if q < 0:return None
    return dsml_unescape_text(tag[p:q])


def tool_text_unescape(s, end):
    """ds4_tool_text_unescape."""
    out = [];i = 0
    while i < len(s):
        if s[i] == '&':
            j = i + 1
            while s.startswith('amp;', j):j += 4
            if s.startswith('lt;', j) and s.startswith(end[1:], j + 3):
                if s.startswith('&amp;', i):out.append('&');i += 5
                else:out.append('<');i += 4
                continue
        out.append(s[i]);i += 1
    return ''.join(out)


def _args_add(args, name, value, is_string):
    piece = (', ' if args else '') + json_escape(name or '') + ': '
    if is_string == 'true':piece += json_escape(value or '')
    else:
        text = value if value is not None else 'null'
        stripped = text.lstrip(SPACE);end = skip_value(stripped, 0)
        minified = json_minify_raw_value(stripped[:end]) if end is not None else text
        piece += minified if minified else 'null'
    args.append(piece)


def _leaf(s, p, param_start, param_end, members):
    if not s.startswith(param_start, p):return None
    tag_end = _find(s, '>', p)
    if tag_end is None:return None
    tag = s[p:tag_end + 1];name = dsml_attr(tag, 'name');is_string = dsml_attr(tag, 'string')
    if name is None:return None
    value_end = _find(s, param_end, tag_end + 1)
    if value_end is None:return None
    value = s[tag_end + 1:value_end];kind = is_string if is_string is not None else 'true'
    if kind == 'true':value = tool_text_unescape(value, param_end)
    _args_add(members, name, value, kind)
    return value_end + len(param_end)


def _nested(s, p, param_start, param_end):
    members = [];any_member = False
    while True:
        p = skip_ws(s, p)
        if not s.startswith(param_start, p):break
        p = _leaf(s, p, param_start, param_end, members)
        if p is None:return None, None
        any_member = True
    if not any_member:return None, None
    return '{' + ''.join(members) + '}', p


def split_reasoning_content(text):
    body = text[7:] if text.startswith('<think>') else text
    end = body.find('</think>')
    if end >= 0:return body[end + 8:], body[:end]
    return text, None


def _cut_c(text):
    """C buffers end at NUL."""
    return text.split('\0', 1)[0]


def parse_deepseek_generated(text, require_thinking_closed):
    """parse_deepseek_generated_message_ex -> (ok, content, reasoning, calls, raw_tool_text)."""
    text = _cut_c(text or '');tool_search = 0;recovered_unclosed_tool = False
    if require_thinking_closed:
        think_end = find_tool_structural_text(text, '</think>', True)
        if think_end is None:
            candidate = find_any_tool_start(text)
            if candidate is None or find_any_tool_end(text, candidate) is None:
                body = text[7:] if text.startswith('<think>') else text
                return True, '', body, [], None
            tool_search = candidate;recovered_unclosed_tool = True
        else:tool_search = think_end + 8
    style = 0
    start = _find(text, '\n\n' + TOOL_CALLS_START, tool_search)
    if start is None:start = _find(text, TOOL_CALLS_START, tool_search)
    if start is None:
        start = _find(text, '\n\n' + TOOL_CALLS_START_SHORT, tool_search);style = 2 if start is not None else style
    if start is None:
        start = _find(text, TOOL_CALLS_START_SHORT, tool_search);style = 2 if start is not None else style
    if start is None:
        start = _find(text, '\n\n<tool_calls>', tool_search);style = 1 if start is not None else style
    if start is None:
        start = _find(text, '<tool_calls>', tool_search);style = 1 if start is not None else style
    v41 = _find(text, '\n\n' + DS41_TOOL_CALLS_START, tool_search)
    if v41 is None:v41 = _find(text, DS41_TOOL_CALLS_START, tool_search)
    if v41 is not None and (start is None or v41 < start):start = v41;style = 3
    if start is None:
        content, reasoning = split_reasoning_content(text)
        return True, content, reasoning, [], None
    content_len = start
    while content_len > 0 and text[content_len - 1] in SPACE:content_len -= 1
    calls_start, calls_end, invoke_start, invoke_end, param_start, param_end = STYLES[style]
    p = _find(text, calls_start, start)
    if p is None:return False, None, None, [], None
    p += len(calls_start);calls = []
    while True:
        p = skip_ws(text, p)
        if text.startswith(calls_end, p):
            raw = text[start:p + len(calls_end)]
            if recovered_unclosed_tool:
                body = text[:content_len]
                if content_len >= 7 and body.startswith('<think>'):body = body[7:]
                return True, '', body, calls, raw
            content, reasoning = split_reasoning_content(text[:content_len])
            return True, content, reasoning, calls, raw
        if not text.startswith(invoke_start, p):return False, None, None, calls, None
        tag_end = _find(text, '>', p)
        if tag_end is None:return False, None, None, calls, None
        name = dsml_attr(text[p:tag_end + 1], 'name')
        if name is None:return False, None, None, calls, None
        p = tag_end + 1;args = []
        while True:
            p = skip_ws(text, p)
            if text.startswith(invoke_end, p):p += len(invoke_end);break
            if not text.startswith(param_start, p):return False, None, None, calls, None
            tag_end = _find(text, '>', p)
            if tag_end is None:return False, None, None, calls, None
            tag = text[p:tag_end + 1];param_name = dsml_attr(tag, 'name');param_is_string = dsml_attr(tag, 'string')
            if param_name is None:return False, None, None, calls, None
            value_start = tag_end + 1
            if param_is_string is None and text.startswith(param_start, skip_ws(text, value_start)):
                nested, nested_p = _nested(text, value_start, param_start, param_end)
                if nested is None:return False, None, None, calls, None
                _args_add(args, param_name, nested or '{}', 'false')
                p = skip_ws(text, nested_p)
                if text.startswith(param_end, p):p += len(param_end)
                continue
            value_end = _find(text, param_end, value_start)
            if value_end is None:return False, None, None, calls, None
            kind = param_is_string if param_is_string is not None else 'true'
            value = text[value_start:value_end]
            if kind == 'true':value = tool_text_unescape(value, param_end)
            _args_add(args, param_name, value, kind)
            p = value_end + len(param_end)
        calls.append(dict(id=None, name=name, arguments='{' + ''.join(args) + '}'))


def try_repair_dsml(s):
    """Only a missing outer wrapper around complete invocations is repaired."""
    if not s:return None
    think_end = find_tool_structural_text(s, '</think>', True)
    scan = think_end + 8 if think_end is not None else 0
    first = find_any_tool_start(s, scan)
    if first is None:return None
    for style in (3, 0, 2, 1):
        if s.startswith(STYLES[style][0], first):break
    else:return None
    tags = STYLES[style];counts = [0] * 6;p = scan
    while p < len(s):
        for k, tag in enumerate(tags):
            if s.startswith(tag, p):counts[k] += 1;p += len(tag);break
        else:p += 1
    tos, toe, ios, ioe, pos, poe = counts
    if ios != ioe or pos != poe or tos == toe or toe > tos:return None
    return s + tags[1] * (tos - toe)


def parse_for_response(text, *, has_tools, saw_tool_start, require_thinking_closed, finish):
    """parse_generated_message_for_response_for_syntax (DeepSeek-V4.1).
    Returns dict(ok, content, reasoning, calls, raw_tool_text, finish, recovered)."""
    ok, content, reasoning, calls, raw = parse_deepseek_generated(text, require_thinking_closed)
    if ok:return dict(ok=True, content=content, reasoning=reasoning, calls=calls, raw_tool_text=raw, finish=finish, recovered=False)
    finish = finish or 'stop';recovered = False
    if has_tools and saw_tool_start and finish != 'error':
        finish = 'length' if finish == 'length' else 'stop';recovered = True
    return dict(ok=False, content=_cut_c(text or ''), reasoning=None, calls=[], raw_tool_text=None, finish=finish, recovered=recovered)


def finish_message(text, *, has_tools, think, finish):
    """The finish-time path of the native chat worker: wrapper repair, then the
    response parse. saw_tool_start/end follow the stream observer's marker scan
    after the closing </think> (or the whole text without thinking)."""
    scan = text
    if think:
        end = text.rfind('</think>')
        scan = text[end + 8:] if end >= 0 else ''
    saw_start = has_tools and find_any_tool_start(scan) is not None
    saw_end = has_tools and saw_start and find_any_tool_end(scan, find_any_tool_start(scan)) is not None
    if has_tools and saw_start and not saw_end and finish not in ('error', 'length'):
        repaired = try_repair_dsml(text)
        if repaired is not None:
            ok, _, _, calls, _ = parse_deepseek_generated(repaired, False)
            if ok and calls:text = repaired
    result = parse_for_response(text, has_tools=has_tools, saw_tool_start=saw_start, require_thinking_closed=think, finish=finish)
    if result['calls']:
        for call in result['calls']:call['id'] = random_tool_id()
        result['finish'] = 'tool_calls'
    return result


def random_tool_id():
    return 'call_' + secrets.token_hex(16)


def json_object_or_empty(arguments):
    """append_json_object_or_empty: a compact JSON object text."""
    args = json_args_parse(arguments)
    if args is None:return '{}'
    return '{' + ','.join(json_escape(k) + ':' + (json_escape(v) if s else v) for k, v, s in args) + '}'


def tool_call_deltas(calls):
    """append_tool_call_deltas_json as Python rows (arguments as the JSON-object string)."""
    return [dict(index=i, id=c['id'], type='function', function=dict(name=c['name'] or '', arguments=json_object_or_empty(c['arguments'])))
            for i, c in enumerate(calls)]
