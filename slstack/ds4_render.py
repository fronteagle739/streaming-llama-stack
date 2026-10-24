"""In-process DeepSeek-V4.1 chat render + tokenizer: a Python port of ds4-win's.

Provenance (exact logic port, no process): engines/ds4-win at stack revision
b5907f2 (fork base 058ae09159a1, merge e1e45374ddc3):
  ds4_server.c  parse_chat_request / parse_messages / parse_openai_content(_object)
                json_content / parse_tool_calls_value / parse_function_call
                parse_tools_value / openai_function_schema_from_tool
                append_responses_namespace_tool_schemas / responses_*_schema_from_tool
                append_tools_prompt_text(v41) / append_dsml_tool_calls_text(v41)
                json_args_parse / json_minify_raw_value / append_dsml_* / append_tool_result_text
                ds41_order_messages / render_deepseek41_chat_ordered (ds4_combo_active=true)
                live_text_prefix_prompt (memory-text rule)
  ds4.c         vocab_load (DeepSeek branch) / special_token_at / tokenize_rendered_chat_vocab
                bpe_tokenize_text (DeepSeek/JoyAI pre-tokenizer) / bpe_emit_piece / byte_encode
                ds4_token_text / vocab_token_is_literal_special
                ds4_deepseek41_reasoning_effort_text / ds4_think_mode_enabled
  ds4_tool_text.h ds4_tool_text_needs_escape
  ds4_quality_phase.h DS41_TOOL_CALLS_START/END
The request arrives as the JSON object the native receives (Native.render posts
json.dumps(body, separators=(',',':'))), so raw JSON values are reproduced with
that same serializer. The GPU gate (tests/combo_render_gate.py) asserts that the
token ids equal the native render byte for byte.
"""
from __future__ import annotations
import heapq
import json
import re
import struct
from pathlib import Path

BOS = '<｜begin▁of▁sentence｜>'
EOS = '<｜end▁of▁sentence｜>'
SYSTEM = '<｜System｜>'
USER = '<｜User｜>'
ASSISTANT = '<｜Assistant｜>'
DS41_TOOL_CALLS_START = '<｜DSML｜ calls>'
DS41_TOOL_CALLS_END = '</｜DSML｜ calls>'
DS41_INVOKE_START = '<｜DSML｜ invoke'
DS41_INVOKE_END = '</｜DSML｜ invoke>'
DS41_PARAM_START = '<｜DSML｜ parameter'
DS41_PARAM_END = '</｜DSML｜ parameter>'
THINK_NONE, THINK_HIGH, THINK_MAX, THINK_LOW, THINK_MEDIUM = 'none', 'high', 'max', 'low', 'medium'


class RenderRefused(ValueError):
    """The native parser would answer 'invalid JSON request' (or a named refusal)."""


# ---------------------------------------------------------------- JSON helpers
def _raw(value):
    """Exact raw text the native sees for a nested value (Native.http serializer)."""
    return json.dumps(value, allow_nan=False, separators=(',', ':'))


def cstr(value):
    """json_string -> C string: lone surrogates become U+FFFD; NUL truncates."""
    if any('\ud800' <= ch <= '\udfff' for ch in value):
        value = ''.join('�' if '\ud800' <= ch <= '\udfff' else ch for ch in value)
    return value.split('\0', 1)[0]


def json_escape(s):
    out = ['"']
    for ch in s:
        c = ord(ch)
        if ch in '"\\':out.append('\\' + ch)
        elif ch == '\n':out.append('\\n')
        elif ch == '\r':out.append('\\r')
        elif ch == '\t':out.append('\\t')
        elif c < 0x20:out.append('\\u%04x' % c)
        else:out.append(ch)
    out.append('"')
    return ''.join(out)


def json_minify_raw_value(text):
    """Whitespace outside strings removed (input is already compact here)."""
    out = [];in_string = escape = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escape:escape = False
            elif ch == '\\':escape = True
            elif ch == '"':in_string = False
        elif ch == '"':in_string = True;out.append(ch)
        elif ch not in ' \t\n\r\v\f':out.append(ch)
    return ''.join(out)


WS = ' \t\n\r\v\f'
_NUMBER = re.compile(r'[+-]?(?:0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?|'
                     r'(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?|[nN][aA][nN])')


def _ws(s, i):
    while i < len(s) and s[i] in WS:i += 1
    return i


def scan_string(s, i):
    """json_string_length on a str: (decoded C string, end) or None."""
    i = _ws(s, i)
    if i >= len(s) or s[i] != '"':return None
    i += 1;out = []
    while i < len(s) and s[i] != '"':
        ch = s[i];i += 1
        if ch != '\\':out.append(ch);continue
        if i >= len(s):return None
        ch = s[i];i += 1
        simple = {'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}
        if ch in simple:out.append(simple[ch]);continue
        if ch != 'u':return None
        def u16(at):
            digits = s[at + 2:at + 6]
            if s[at:at + 2] != '\\u' or len(digits) < 4 or any(d not in '0123456789abcdefABCDEF' for d in digits):return None
            return int(digits, 16)
        cp = u16(i - 2)
        if cp is None:return None
        i += 4
        if 0xd800 <= cp <= 0xdbff:
            lo = u16(i)
            if lo is not None and 0xdc00 <= lo <= 0xdfff:cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);i += 6
            else:cp = 0xfffd
        if cp > 0x10ffff or 0xd800 <= cp <= 0xdfff:cp = 0xfffd
        out.append(chr(cp))
    if i >= len(s):return None
    return cstr(''.join(out)), i + 1


def skip_value(s, i, depth=0):
    """json_skip_value_depth: end index or None."""
    i = _ws(s, i)
    if i >= len(s):return None
    ch = s[i]
    if ch == '"':
        hit = scan_string(s, i);return hit[1] if hit else None
    if ch in '{[':
        if depth >= 256:return None
        close = '}' if ch == '{' else ']';i = _ws(s, i + 1)
        if i < len(s) and s[i] == close:return i + 1
        while True:
            if close == '}':
                hit = scan_string(s, i)
                if not hit:return None
                i = _ws(s, hit[1])
                if i >= len(s) or s[i] != ':':return None
                i += 1
            i = skip_value(s, i, depth + 1)
            if i is None:return None
            i = _ws(s, i)
            if i < len(s) and s[i] == close:return i + 1
            if i >= len(s) or s[i] != ',':return None
            i += 1
    for literal in ('true', 'false', 'null'):
        if s.startswith(literal, i):return i + len(literal)
    match = _NUMBER.match(s, i)
    return match.end() if match else None


def json_args_parse(arguments):
    """json_args_parse: top-level members in order as (key, value, is_string), or None."""
    s = arguments if arguments is not None else ''
    i = _ws(s, 0)
    if i >= len(s) or s[i] != '{':return None
    i = _ws(s, i + 1);out = []
    while i < len(s) and s[i] != '}':
        hit = scan_string(s, i)
        if not hit:return None
        key, i = hit;i = _ws(s, i)
        if i >= len(s) or s[i] != ':':return None
        i = _ws(s, i + 1)
        if i < len(s) and s[i] == '"':
            hit = scan_string(s, i)
            if not hit:return None
            value, i = hit;out.append((key, value, True))
        else:
            end = skip_value(s, i)
            if end is None:return None
            out.append((key, json_minify_raw_value(s[i:end]), False));i = end
        i = _ws(s, i)
        if i < len(s) and s[i] == ',':i += 1
        i = _ws(s, i)
    if i >= len(s):return None
    return out


# ---------------------------------------------------------------- request parse
def _content(value):
    """json_content: string / null -> '' / array of strings and {text} parts / other -> ''."""
    if isinstance(value, str):return cstr(value)
    if value is None:return ''
    if not isinstance(value, list):return ''
    parts = []
    for item in value:
        if isinstance(item, str):parts.append(cstr(item))
        elif isinstance(item, dict):
            if 'text' in item:
                if not isinstance(item['text'], str):raise RenderRefused('invalid JSON request')
                parts.append(cstr(item['text']))
    return ''.join(parts)


def _openai_content(value):
    if isinstance(value, str):return cstr(value)
    if value is None:return None
    if not isinstance(value, list):raise RenderRefused('invalid JSON request')
    parts = []
    for item in value:
        if isinstance(item, str):parts.append(cstr(item))
        elif isinstance(item, dict):
            kind = item.get('type')
            if 'type' in item and not isinstance(kind, str):raise RenderRefused('invalid JSON request')
            text = _content(item['text']) if 'text' in item else None
            if 'image_url' in item:
                url = item['image_url']
                if not (isinstance(url, str) or (isinstance(url, dict) and isinstance(url.get('url'), str))):
                    raise RenderRefused('invalid JSON request')
            if kind in ('image_url', 'input_image'):
                raise RenderRefused('mode1 combined contract requires text-only native state')
            if text is not None:parts.append(text)
    return ''.join(parts) if parts else ''


def _tool_calls(value):
    if value is None:return []
    if not isinstance(value, list):raise RenderRefused('invalid JSON request')
    calls = []
    for item in value:
        if not isinstance(item, dict):raise RenderRefused('invalid JSON request')
        call = dict(id=None, name=None, arguments=None)
        for key, member in item.items():
            if key == 'id':
                if not isinstance(member, str):raise RenderRefused('invalid JSON request')
                call['id'] = cstr(member)
            elif key == 'function':
                if not isinstance(member, dict):raise RenderRefused('invalid JSON request')
                for fkey, fvalue in member.items():
                    if fkey == 'name':
                        if not isinstance(fvalue, str):raise RenderRefused('invalid JSON request')
                        call['name'] = cstr(fvalue)
                    elif fkey == 'arguments':
                        call['arguments'] = cstr(fvalue) if isinstance(fvalue, str) else _raw(fvalue)
        if call['name'] is not None and call['arguments'] is not None:calls.append(call)
    return calls


def parse_messages(value):
    if not isinstance(value, list):raise RenderRefused('invalid JSON request')
    messages = []
    for item in value:
        if not isinstance(item, dict):raise RenderRefused('invalid JSON request')
        msg = dict(role=None, content=None, reasoning=None, tool_call_id=None, tool_call_ids=[], calls=[], raw_tool_text=None)
        for key, member in item.items():
            if key == 'role':
                if not isinstance(member, str):raise RenderRefused('invalid JSON request')
                msg['role'] = cstr(member)
            elif key == 'content':msg['content'] = _openai_content(member)
            elif key == 'reasoning_content':msg['reasoning'] = _content(member)
            elif key == 'tool_call_id':
                if not isinstance(member, str):raise RenderRefused('invalid JSON request')
                ident = cstr(member)
                if ident:
                    if msg['tool_call_id'] is None:msg['tool_call_id'] = ident
                    if ident not in msg['tool_call_ids']:msg['tool_call_ids'].append(ident)
            elif key == 'tool_calls':msg['calls'] = _tool_calls(member)
        if msg['role'] is None:msg['role'] = 'user'
        if msg['content'] is None:msg['content'] = ''
        messages.append(msg)
    return messages


def _namespace_schema(tool, namespace):
    if not isinstance(tool, dict):return None
    kind = name = description = parameters = None
    for key, member in tool.items():
        if key == 'type':
            if not isinstance(member, str):return None
            kind = cstr(member)
        elif key == 'name':
            if not isinstance(member, str):return None
            name = cstr(member)
        elif key == 'description':
            if not isinstance(member, str):return None
            description = cstr(member)
        elif key in ('parameters', 'input_schema'):parameters = _raw(member)
    if (kind is None or kind == 'function') and namespace is not None and name:
        return ('{"name":' + json_escape(namespace + name) + ',"description":' + json_escape(description or '') +
                ',"parameters":' + (parameters if parameters is not None else '{"type":"object","properties":{}}') + '}')
    return None


def parse_tools(value):
    """parse_tools_value: one raw function schema per line (DS4 prompt shape)."""
    if value is None:return ''
    if not isinstance(value, list):raise RenderRefused('invalid JSON request')
    lines = []
    for tool in value:
        raw = _raw(tool)
        if isinstance(tool, dict) and 'function' in tool:
            lines.append(_raw(tool['function']))
            continue
        appended = False
        if isinstance(tool, dict):
            kind = name = tools = None;ok = True
            for key, member in tool.items():
                if key == 'type':
                    if not isinstance(member, str):ok = False;break
                    kind = cstr(member)
                elif key == 'name':
                    if not isinstance(member, str):ok = False;break
                    name = cstr(member)
                elif key == 'tools':tools = member
            if ok and kind == 'namespace' and name is not None and tools is not None and isinstance(tools, list):
                for item in tools:
                    schema = _namespace_schema(item, name)
                    if schema:lines.append(schema);appended = True
        if appended:continue
        special = None
        if isinstance(tool, dict):
            kind = description = parameters = None;ok = True
            for key, member in tool.items():
                if key == 'type':
                    if not isinstance(member, str):ok = False;break
                    kind = cstr(member)
                elif key == 'description':
                    if not isinstance(member, str):ok = False;break
                    description = cstr(member)
                elif key == 'parameters':parameters = _raw(member)
            if ok and kind == 'tool_search':
                special = ('{"name":"tool_search","description":' + json_escape(description if description is not None else 'Search available tools.') +
                           ',"parameters":' + (parameters if parameters is not None else '{"type":"object","properties":{}}') + '}')
        lines.append(special if special else raw)
    return '\n'.join(line for line in lines if line)


EFFORTS = {'max': THINK_MAX, 'xhigh': THINK_HIGH, 'high': THINK_HIGH, 'medium': THINK_MEDIUM,
           'low': THINK_LOW, 'minimal': THINK_LOW, 'none': THINK_NONE}
THINK_OFF_ALIASES = {'deepseek-chat', 'qwen3.8-flash-next-chat', 'qwen3.8-flash-next-no-think', 'qwen3.8-flash-next-nothink',
    'qwen/qwen3.8-flash-next-chat', 'glm-5.2-chat', 'glm-5.2-no-think', 'glm-5.2-nothink', 'zai/glm-5.2-chat',
    'glm-5.3-flash-chat', 'glm-5.3-flash-no-think', 'glm-5.3-flash-nothink', 'zai/glm-5.3-flash-chat'}
THINK_ON_ALIASES = {'deepseek-reasoner', 'qwen3.8-flash-next-reasoner', 'qwen/qwen3.8-flash-next-reasoner', 'glm-5.2-reasoner',
    'zai/glm-5.2-reasoner', 'glm-5.3-flash-reasoner', 'zai/glm-5.3-flash-reasoner'}


def _effort(value, current):
    if value is None:return current
    if not isinstance(value, str) or cstr(value) not in EFFORTS:raise RenderRefused('invalid JSON request')
    return EFFORTS[cstr(value)]


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):raise RenderRefused('invalid JSON request')
    return value


def parse_stop(value):
    """ds4-win parse_stop: a string or an array whose string members are the stop
    sequences (other members skipped); any other value is ignored; empty strings dropped."""
    if isinstance(value, str):values = [value]
    elif isinstance(value, list):values = [v for v in value if isinstance(v, str)]
    else:return []
    return [v for v in (cstr(v) for v in values) if v]


def parse_request(body):
    """parse_chat_request for the DeepSeek-V4.1 syntax. Returns the fields that
    decide rendering and the reply protocol."""
    if not isinstance(body, dict):raise RenderRefused('invalid JSON request')
    messages = None;tools = '';tool_choice_none = False;model = None;max_tokens = None
    got_thinking = False;thinking = True;effort = THINK_HIGH;stream = False;include_usage = False;stops = []
    for key, value in body.items():
        if key == 'messages':messages = parse_messages(value)
        elif key == 'tools':tools = parse_tools(value)
        elif key == 'tool_choice':tool_choice_none = isinstance(value, str) and cstr(value) == 'none'
        elif key == 'model':
            if not isinstance(value, str):raise RenderRefused('invalid JSON request')
            model = cstr(value)
        elif key in ('max_tokens', 'max_completion_tokens'):
            number = _number(value);max_tokens = 0 if not number >= 0 else int(min(number, 2**31 - 1))
        elif key in ('temperature', 'top_p', 'min_p', 'top_k', 'seed'):_number(value)
        elif key == 'stop':stops = parse_stop(value)
        elif key == 'stream':
            if not isinstance(value, bool):raise RenderRefused('invalid JSON request')
            stream = value
        elif key == 'stream_options':
            if isinstance(value, dict) and isinstance(value.get('include_usage'), bool):include_usage = value['include_usage']
        elif key == 'thinking':
            if isinstance(value, bool):thinking = value
            elif isinstance(value, dict):
                kind = value.get('type')
                if 'type' in value and not isinstance(kind, str):raise RenderRefused('invalid JSON request')
                if kind == 'enabled':thinking = True
                elif kind == 'disabled':thinking = False
            got_thinking = True
        elif key == 'reasoning_effort':effort = _effort(value, effort)
        elif key == 'chat_template_kwargs':
            if isinstance(value, dict):
                for name, member in value.items():
                    if name == 'enable_thinking':
                        if not isinstance(member, bool):raise RenderRefused('invalid JSON request')
                        thinking = member;got_thinking = True
                    elif name == 'reasoning_effort':effort = _effort(member, effort)
        elif key == 'think':
            if not isinstance(value, bool):raise RenderRefused('invalid JSON request')
            thinking = value;got_thinking = True
    if messages is None:raise RenderRefused('missing messages')
    if not got_thinking and model in THINK_OFF_ALIASES:thinking = False
    if not got_thinking and model in THINK_ON_ALIASES:thinking = True
    think_mode = THINK_NONE if not thinking or effort == THINK_NONE else effort
    has_tools = bool(tools) and not tool_choice_none
    return dict(messages=messages, tool_schemas=tools if has_tools else '', has_tools=has_tools, think_mode=think_mode,
                model=model, max_tokens=max_tokens, stream=stream, include_usage=include_usage, stops=stops)


# ---------------------------------------------------------------- DS4.1 render
def tool_text_needs_escape(s, i, end):
    if s.startswith(end, i):return True
    if s[i] != '&':return False
    j = i + 1
    while s.startswith('amp;', j):j += 4
    return s.startswith('lt;', j) and s.startswith(end[1:], j + 3)


def escape_tool_text(s, end):
    out = []
    for i, ch in enumerate(s):
        if tool_text_needs_escape(s, i, end):out.append('&lt;' if ch == '<' else '&amp;')
        else:out.append(ch)
    return ''.join(out)


def dsml_attr_escaped(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def dsml_json_literal(s):
    out = [];i = 0
    while i < len(s):
        if s.startswith(DS41_PARAM_END, i):out.append('\\u003c')
        else:out.append(s[i])
        i += 1
    return ''.join(out)


def dsml_tool_calls_text(calls, raw_tool_text=None):
    if not calls:return ''
    if raw_tool_text:return raw_tool_text
    out = ['\n\n' + DS41_TOOL_CALLS_START + '\n']
    for call in calls:
        out.append(DS41_INVOKE_START + ' name="' + dsml_attr_escaped(call['name']) + '">\n')
        args = json_args_parse(call['arguments'])
        if args is None:
            out.append(DS41_PARAM_START + ' name="arguments" string="true">' + escape_tool_text(call['arguments'], DS41_PARAM_END) + DS41_PARAM_END + '\n')
        else:
            for key, value, is_string in args:
                out.append(DS41_PARAM_START + ' name="' + dsml_attr_escaped(key) + '" string="' + ('true' if is_string else 'false') + '">')
                out.append(escape_tool_text(value, DS41_PARAM_END) if is_string else dsml_json_literal(value))
                out.append(DS41_PARAM_END + '\n')
        out.append(DS41_INVOKE_END + '\n')
    out.append(DS41_TOOL_CALLS_END)
    return ''.join(out)


def tools_prompt_text(tool_schemas):
    if not tool_schemas:return ''
    return ('## Tools\n\nYou can invoke tools using this format:\n\n' + DS41_TOOL_CALLS_START + '\n' +
            DS41_INVOKE_START + ' name="$TOOL_NAME">\n' +
            DS41_PARAM_START + ' name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE' + DS41_PARAM_END + '\n' +
            DS41_INVOKE_END + '\n' + DS41_TOOL_CALLS_END + '\n\n' +
            'String values use string="true"; all other values use JSON and string="false". '
            'Inside string values only, escape a literal </｜DSML｜ parameter> as &lt;/｜DSML｜ parameter>. '
            'To write that escaped spelling literally, use &amp;lt;/｜DSML｜ parameter>. Other HTML entities are unchanged.\n\n'
            'Finish reasoning with </think> before tool calls or a final response.\n\n'
            '### Available Tool Schemas\n\n' + tool_schemas +
            '\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. '
            'Use the exact parameter names from the schemas.')


def effort_text(mode):
    level = {THINK_LOW: 50, THINK_HIGH: 75, THINK_MAX: 100}.get(mode, -1)
    if level <= 0:return None
    return 'Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n' % level


def _system(role):return role in ('system', 'developer')
def _user_like(role):return role in ('user', 'tool', 'function')


def uses_tool_context(messages, tool_schemas):
    if tool_schemas:return True
    return any((m['role'] == 'assistant' and m['calls']) or m['role'] in ('tool', 'function') for m in messages)


def order_messages(messages):
    ordered = list(messages);order = {};i = 0
    while i < len(messages):
        m = messages[i]
        if m['role'] == 'assistant' and m['calls']:
            order = {}
            for j, call in enumerate(m['calls']):
                if call.get('id'):order[call['id']] = j
        if not _user_like(m['role']):i += 1;continue
        begin = i;results = []
        while i < len(messages) and _user_like(messages[i]['role']):
            if messages[i]['role'] != 'user':
                ident = messages[i]['tool_call_id']
                results.append((order.get(ident, 0) if ident is not None else 0, i, messages[i]))
            i += 1
        results.sort(key=lambda row: (row[0], row[1]))
        it = iter(results)
        for j in range(begin, i):
            if messages[j]['role'] != 'user':ordered[j] = next(it)[2]
    return ordered


def render_deepseek41(messages, tool_schemas, think_mode, *, combo_active=True, tool_memory=None):
    """render_deepseek41_chat_ordered(start=0, live_tail=false). tool_memory maps
    call id -> exact DSML block (tool_memory_attach_to_messages); a fresh native has none."""
    ordered = order_messages(messages)
    think = think_mode != THINK_NONE
    keep_reasoning = uses_tool_context(messages, tool_schemas) or combo_active
    effort = effort_text(think_mode)
    last_user = -1
    for i, m in enumerate(messages):
        if _user_like(m['role']) or (i > 0 and _system(m['role'])):last_user = i
    out = [BOS];initial_system = pending = user_open = False
    if effort or tool_schemas:
        out.append(SYSTEM);initial_system = True
        if effort:out.append(effort)
        if tool_schemas:out.append(tools_prompt_text(tool_schemas))
    for i, m in enumerate(ordered):
        if _system(m['role']):
            if i == 0 and initial_system:
                if tool_schemas:out.append('\n\n')
            else:out.append(SYSTEM)
            out.append(m['content']);pending = i > 0;user_open = False
        elif _user_like(m['role']):
            out.append('\n\n' if user_open else USER)
            if m['role'] != 'user':out.append('<tool_result>' + escape_tool_text(m['content'], '</tool_result>') + '</tool_result>')
            else:out.append(m['content'])
            pending = user_open = True
        elif m['role'] == 'assistant':
            if pending:
                out.append(ASSISTANT)
                if think and (keep_reasoning or i > last_user):out.append('<think>' + (m['reasoning'] or ''))
                out.append('</think>')
            raw = None
            if tool_memory and m['calls']:
                blocks = [tool_memory.get(c.get('id')) for c in m['calls']]
                if all(blocks) and len(set(blocks)) == 1:raw = blocks[0]
            out.append(m['content'] + dsml_tool_calls_text(m['calls'], raw) + EOS)
            pending = user_open = False
    if pending:out.append(ASSISTANT + ('<think>' if think else '</think>'))
    return ''.join(out)


# ---------------------------------------------------------------- tokenizer
def _gpt2_bytes():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(0xa1, 0xad)) + list(range(0xae, 0x100))
    cs = bs[:];n = 0
    for b in range(256):
        if b not in bs:bs.append(b);cs.append(256 + n);n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR = _gpt2_bytes()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


def gguf_tokenizer(path):
    """tokenizer.ggml.tokens and .merges from the GGUF header (no tensor data read)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    found = {}
    with Path(path).open('rb') as f:
        magic, version = f.read(4), struct.unpack('<I', f.read(4))[0]
        if magic != b'GGUF' or version < 2:raise ValueError('Not a GGUF v2+ model')
        _tensors, count = struct.unpack('<QQ', f.read(16))
        def string():return f.read(struct.unpack('<Q', f.read(8))[0])
        def skip(kind):
            if kind == 8:string()
            elif kind == 9:
                inner, n = struct.unpack('<IQ', f.read(12))
                if inner == 8:
                    for _ in range(n):string()
                elif inner == 9:
                    for _ in range(n):skip(9)
                else:f.seek(sizes[inner] * n, 1)
            else:f.seek(sizes[kind], 1)
        for _ in range(count):
            key = string().decode();kind = struct.unpack('<I', f.read(4))[0]
            if key in ('tokenizer.ggml.tokens', 'tokenizer.ggml.merges') and kind == 9:
                inner, n = struct.unpack('<IQ', f.read(12))
                if inner != 8:raise ValueError(key + ' is not a string array')
                found[key] = [string().decode('utf-8', 'surrogateescape') for _ in range(n)]
            else:skip(kind)
            if len(found) == 2:break
    if len(found) != 2:raise ValueError('GGUF tokenizer token/merge table is missing')
    return found['tokenizer.ggml.tokens'], found['tokenizer.ggml.merges']


def _alpha(c):return 65 <= c <= 90 or 97 <= c <= 122
def _digit(c):return 48 <= c <= 57
def _space(c):return c in (32, 9, 10, 13, 11, 12)
def _newline(c):return c in (10, 13)
def _punct(c):return 33 <= c <= 47 or 58 <= c <= 64 or 91 <= c <= 96 or 123 <= c <= 126


def _ulen(c):
    if c < 0x80:return 1
    if c & 0xe0 == 0xc0:return 2
    if c & 0xf0 == 0xe0:return 3
    if c & 0xf8 == 0xf0:return 4
    return 1


def _next(s, pos):
    n = _ulen(s[pos])
    return pos + (n if pos + n <= len(s) else 1)


def _letter(s, pos):
    c = s[pos]
    return _alpha(c) if c < 128 else True


def _cjk(s, pos):
    c0 = s[pos]
    if c0 < 128:return False
    n = _ulen(c0)
    if pos + n > len(s):n = 1
    if n == 1:cp = c0
    elif n == 2:cp = (c0 & 0x1f) << 6 | (s[pos + 1] & 0x3f)
    elif n == 3:cp = (c0 & 0x0f) << 12 | (s[pos + 1] & 0x3f) << 6 | (s[pos + 2] & 0x3f)
    else:cp = (c0 & 0x07) << 18 | (s[pos + 1] & 0x3f) << 12 | (s[pos + 2] & 0x3f) << 6 | (s[pos + 3] & 0x3f)
    return 0x4e00 <= cp <= 0x9fa5 or 0x3040 <= cp <= 0x309f or 0x30a0 <= cp <= 0x30ff


def _letters(s, pos):
    while pos < len(s) and _letter(s, pos):pos = _next(s, pos)
    return pos


def pretokenize(s):
    """bpe_tokenize_text for the DeepSeek family (JoyAI-style split), on UTF-8 bytes."""
    n = len(s);pos = 0
    while pos < n:
        start = pos;c = s[pos]
        if _digit(c):
            k = 0
            while pos < n and _digit(s[pos]) and k < 3:pos += 1;k += 1
        elif _cjk(s, pos):
            pos = _next(s, pos)
            while pos < n and _cjk(s, pos):pos = _next(s, pos)
        elif _punct(c) and pos + 1 < n and _alpha(s[pos + 1]):
            pos += 1
            while pos < n and _alpha(s[pos]):pos += 1
        elif _letter(s, pos):pos = _letters(s, pos)
        elif not _newline(c) and not _punct(c) and pos + 1 < n and _letter(s, pos + 1):pos = _letters(s, pos + 1)
        elif c == 32 and pos + 1 < n and _punct(s[pos + 1]):
            pos += 1
            while pos < n and _punct(s[pos]):pos += 1
            while pos < n and _newline(s[pos]):pos += 1
        elif _punct(c):
            while pos < n and _punct(s[pos]):pos += 1
            while pos < n and _newline(s[pos]):pos += 1
        elif _space(c):
            p = pos;last_newline_end = 0
            while p < n and _space(s[p]):
                sc = s[p];p += 1
                if _newline(sc):last_newline_end = p
            if last_newline_end:pos = last_newline_end
            elif p < n and p > pos + 1 and (_letter(s, p) or _punct(s[p])):pos = p - 1
            else:pos = p
        else:pos = _next(s, pos)
        if pos == start:pos = _next(s, pos)
        yield s[start:pos]


class Tokenizer:
    def __init__(self, tokens, merges):
        self.tokens = tokens
        self.ids = {}
        for i, text in enumerate(tokens):self.ids.setdefault(text, i)
        self.ranks = {}
        for i, merge in enumerate(merges):self.ranks.setdefault(merge, i)
        def lookup(text):
            if text not in self.ids:raise ValueError('ds4: required tokenizer token is missing: ' + text)
            return self.ids[text]
        self.bos, self.eos, self.system, self.user, self.assistant = (lookup(t) for t in (BOS, EOS, SYSTEM, USER, ASSISTANT))
        self.think_start, self.think_end, self.dsml = lookup('<think>'), lookup('</think>'), lookup('｜DSML｜')
        # special_token_at, DeepSeek-V4.1 family, in table order (absent ids skipped).
        self.specials = [(BOS, self.bos), (EOS, self.eos), ('[gMASK]', self.bos), ('<|system|>', self.system),
            (SYSTEM, self.system), (USER, self.user), (ASSISTANT, self.assistant), ('<|user|>', self.user),
            ('<|assistant|>', self.assistant), ('<think>', self.think_start), ('</think>', self.think_end), ('｜DSML｜', self.dsml)]
        self.first = {text[0] for text, _ in self.specials}
        self.cache = {}

    @classmethod
    def from_gguf(cls, path):return cls(*gguf_tokenizer(path))

    def piece(self, raw):
        hit = self.cache.get(raw)
        if hit is not None:return hit
        symbols = [BYTE_TO_CHAR[b] for b in raw]
        # Greedy lowest-rank merge, ties to the leftmost pair (bpe_emit_piece), via a heap.
        n = len(symbols);prev = list(range(-1, n - 1));nxt = list(range(1, n + 1));alive = [True] * n
        heap = []
        def push(i):
            j = nxt[i]
            if j < n:
                rank = self.ranks.get(symbols[i] + ' ' + symbols[j])
                if rank is not None:heapq.heappush(heap, (rank, i, symbols[i], symbols[j]))
        for i in range(n - 1):push(i)
        while heap:
            rank, i, left, right = heapq.heappop(heap)
            if not alive[i] or symbols[i] != left:continue
            j = nxt[i]
            if j >= n or symbols[j] != right:continue
            symbols[i] = left + right;alive[j] = False;nxt[i] = nxt[j]
            if nxt[j] < n:prev[nxt[j]] = i
            if prev[i] >= 0:push(prev[i])
            push(i)
        out = []
        for i in range(n):
            if not alive[i]:continue
            ident = self.ids.get(symbols[i])
            if ident is not None:out.append(ident)
            else:
                for b in symbols[i].encode('utf-8'):
                    if b < 0x80 and chr(b) in self.ids:out.append(self.ids[chr(b)])
        if len(raw) <= 256:self.cache[raw] = out
        return out

    def text(self, text):
        out = []
        for piece in pretokenize(text.encode('utf-8', 'surrogateescape')):out.extend(self.piece(piece))
        return out

    def rendered(self, text):
        """tokenize_rendered_chat_vocab: special markers become their ids, the spans between BPE."""
        out = [];span = 0;i = 0;n = len(text)
        while i < n:
            if text[i] in self.first:
                for marker, ident in self.specials:
                    if text.startswith(marker, i):
                        if i > span:out.extend(self.text(text[span:i]))
                        out.append(ident);i += len(marker);span = i
                        break
                else:i += 1
                continue
            i += 1
        if n > span:out.extend(self.text(text[span:]))
        return out

    def token_bytes(self, token):
        """ds4_token_text: literal specials (contain U+FF5C) verbatim, others byte-decoded."""
        if not 0 <= token < len(self.tokens):return b''
        text = self.tokens[token]
        if '｜' in text:return text.encode('utf-8', 'surrogateescape')
        return bytes(CHAR_TO_BYTE[ch] for ch in text if ch in CHAR_TO_BYTE)

    def detokenize(self, tokens):return b''.join(self.token_bytes(t) for t in tokens)


def render(body, tokenizer, *, tool_memory=None):
    """Full canonical render of an OpenAI chat body: (prompt_text, token ids, parsed request)."""
    request = parse_request(body)
    text = render_deepseek41(request['messages'], request['tool_schemas'], request['think_mode'], tool_memory=tool_memory)
    return text, tokenizer.rendered(text), request


def continue_live(live, text, canonical, tokenizer):
    """The native /decode/render prefix rule for a resident session.

    live: token ids already in the engine. Returns (tokens, source): the full
    prompt as the engine should hold it, or (None, 'canonical-miss') when
    neither the ids nor the bytes of the live frontier prefix the new render."""
    if not live:return canonical, 'empty'
    if canonical[:len(live)] == live:return canonical, 'memory-token'
    prefix = tokenizer.detokenize(live);data = text.encode('utf-8')
    if len(data) >= len(prefix) and data[:len(prefix)] == prefix:
        suffix = data[len(prefix):].decode('utf-8', 'surrogateescape')
        return list(live) + tokenizer.rendered(suffix), 'memory-text'
    return None, 'canonical-miss'


def continue_cut_turn(live, turn_start, text, message, tokenizer, *, think, reasoning, content):
    """A resident turn cut short (client stop sequence or disconnect): the producer holds
    every evaluated token of it, the client replays only the part it was sent, so neither
    prefix rule matches. When the client's assistant message is exactly the head of what
    was evaluated (reasoning/content as parsed from live[turn_start:]), the turn is
    remembered as those exact token ids: the engine keeps live and appends only what the
    template adds after the turn ('</think>' if the cut fell inside the reasoning, then
    EOS and the new messages). Returns the full token list, or None."""
    if message is None or message.get('calls'):return None
    mr = message.get('reasoning') or '';mc = message.get('content') or ''
    if not (reasoning.startswith(mr) and content.startswith(mc)):return None
    if think and mc and mr != reasoning:return None
    data = text.encode('utf-8', 'surrogateescape');head = tokenizer.detokenize(live[:turn_start])
    if data[:len(head)] != head:return None
    end = data.find(EOS.encode('utf-8'), len(head))
    if end < 0:return None
    client = ((mr + '</think>' + mc) if think else mc).encode('utf-8', 'surrogateescape')
    if data[len(head):end] != client:return None   # the cut message is exactly the turn rendered after the live head
    generated = tokenizer.detokenize(live[turn_start:])
    closing = b'</think>' if think and b'</think>' not in generated else b''
    return list(live) + tokenizer.rendered((closing + data[end:]).decode('utf-8', 'surrogateescape'))
