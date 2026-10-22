# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_compile_caption.py
# Machine fallback paths cleared; no model data included.
import json
import sys
import urllib.request
ENDPOINT = ''
MODEL = 'deepseek-v4-flash'
SPEC = {'t2v': '', 'i2v': ''}
mode, user_prompt, out_path = (sys.argv[1], sys.argv[2], sys.argv[3])
spec = open(SPEC[mode], encoding='utf-8').read()
print(f'[compile] mode={mode} spec={len(spec)} chars, prompt={user_prompt!r}', flush=True)
payload = {'model': MODEL, 'messages': [{'role': 'system', 'content': spec}, {'role': 'user', 'content': user_prompt}], 'temperature': 0.3, 'max_tokens': 16000}
req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=3600) as r:
    body = json.loads(r.read().decode())
text = (body['choices'][0]['message'].get('content') or '').strip()
usage = body.get('usage', {})
print(f"[compile] reply {len(text)} chars; tokens prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}", flush=True)
if text.startswith('```'):
    text = text.split('```', 2)[1]
    if text.startswith('json'):
        text = text[4:]
    text = text.rsplit('```', 1)[0]
start = text.find('{')
if start > 0:
    text = text[start:]
try:
    obj, _end = json.JSONDecoder().raw_decode(text)
except Exception as exc:
    open(out_path + '.raw', 'w', encoding='utf-8').write(text)
    print(f'[compile] FAIL: reply is not valid JSON ({exc}); raw saved to {out_path}.raw')
    sys.exit(1)
required = {'global_layer', 'dynamic_layer'} if mode == 't2v' else set()
missing = required - set(obj)
if missing:
    print(f'[compile] FAIL: missing required top-level keys {sorted(missing)}; got {sorted(obj)}')
    sys.exit(2)
open(out_path, 'w', encoding='utf-8').write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
segs = obj.get('dynamic_layer', {}).get('timeline_segments', [])
subs = obj.get('global_layer', {}).get('alive_subjects_static', [])
print(f'[compile] OK -> {out_path}: keys={sorted(obj)} segments={len(segs)} subjects={len(subs)}')
