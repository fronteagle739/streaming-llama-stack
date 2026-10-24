"""Per-preset history policy, shared by plain chat and tool rounds.

An absent field retains the existing model-specific behavior. Explicit False
removes previous-turn reasoning only; current tool-round reasoning stays live.
"""
import re

_DSV41 = re.compile(r'(?:deepseek|dsv|ds)[-_ ]*(?:v)?4[._-]?1(?!\d)', re.I)
_THINK = re.compile(r'<think>.*?</think>', re.DOTALL)


def include(preset, model=''):
    value = (preset or {}).get('include_previous_reasoning')
    if value is not None:
        if not isinstance(value, bool):
            raise ValueError('include_previous_reasoning must be a boolean')
        return value
    identifier = str(model or '').lower()
    return ('kimi' in identifier and 'k2.7' in identifier) or bool(_DSV41.search(identifier))


def apply(body, preset):
    value = (preset or {}).get('include_previous_reasoning')
    if value is None:
        return
    include(preset)  # Strictly validate explicit values, including false.
    if value:
        return
    messages = body.get('messages') or []
    last_user = max((i for i, message in enumerate(messages) if message.get('role') == 'user'),
                    default=len(messages))
    rewritten = []
    for index, message in enumerate(messages):
        if index < last_user and message.get('role') == 'assistant':
            message = dict(message)
            for key in ('reasoning_content', 'reasoning', 'thinking', 'reasoning_text'):
                message.pop(key, None)
            if isinstance(message.get('content'), str):
                message['content'] = _THINK.sub('', message['content'])
        rewritten.append(message)
    body['messages'] = rewritten
