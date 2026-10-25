# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/curate_dataset.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse, os, glob, json, base64, io, shutil, re
IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
VID_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif'}

def _b64_jpeg(arr) -> str:
    from PIL import Image
    import numpy as np
    im = Image.fromarray(np.asarray(arr).astype('uint8')).convert('RGB')
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def media_frames(path: str, n_frames: int):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXT:
        from PIL import Image
        import numpy as np
        return ([_b64_jpeg(np.array(Image.open(path).convert('RGB')))], False)
    import imageio.v3 as iio
    import numpy as np
    frames = [np.asarray(f) for f in iio.imiter(path)]
    if not frames:
        return ([], True)
    idx = [int(i * (len(frames) - 1) / max(n_frames - 1, 1)) for i in range(min(n_frames, len(frames)))]
    return ([_b64_jpeg(frames[i]) for i in idx], True)

def _prompt(goal: str, example: str, is_video: bool, task: str) -> str:
    kind = 'video (shown as sampled frames in order)' if is_video else 'image'
    if task == 'caption':
        p = f'''You are captioning a {kind} to build a fine-tuning dataset for a generative video model. The training goal is to learn: "{goal}". Write ONE caption describing this {kind}'s visible content, subject, motion, camera, and style — concrete and specific, no preamble or quotes.'''
        if example:
            p += f'\n\nMatch the STYLE and format of these example captions:\n{example}'
        return p
    return f'Rate from 0.0 to 1.0 how RELEVANT this {kind} is for learning: "{goal}". Consider subject, motion, and style match. Reply with ONLY the number.'

def caption_openai(base_url, model, key, frames_b64, prompt, max_tokens=300):
    import requests
    content = [{'type': 'text', 'text': prompt}]
    for b in frames_b64:
        content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b}'}})
    r = requests.post(base_url.rstrip('/') + '/chat/completions', headers={'Authorization': f"Bearer {key or 'none'}", 'Content-Type': 'application/json'}, json={'model': model, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens, 'temperature': 0.2}, timeout=180)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content'].strip()
_GEMMA_SYS = "You are a terse image captioner for a training dataset. Output ONLY one caption sentence. Never write 'The user', 'Plan', 'Image analysis', or any reasoning. Just the sentence."
_GEMMA_SHOTS = [('assistant', 'A young woman with short blonde hair and green eyes wearing a red dress stands in a sunlit kitchen, photorealistic style.'), ('assistant', 'A muscular man with a shaved head and brown eyes in black armor crouches on a rooftop at night, comic-book style.')]
import re as _re
_GEMMA_THINK = _re.compile("^(the user|plan|image analysis|thinking|here'?s|i must|i need|i'?ll|i will|okay|let'?s|let me|first|step \\d|\\d\\.|\\*|>)", _re.I)
_GEMMA_SENT = _re.compile('((?:A|An|The)\\s+[^.\\n]{25,200}(?:style|photo|render|art|illustration|background)[^.\\n]{0,40}\\.)', _re.I)

def caption_gemma(base_url, model, key, frames_b64, prompt, max_tokens=150):
    import requests
    q = prompt
    imgs = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b}'}} for b in frames_b64]

    def _ask(temp):
        msgs = [{'role': 'system', 'content': _GEMMA_SYS}]
        for _role, _txt in _GEMMA_SHOTS:
            msgs.append({'role': 'user', 'content': q})
            msgs.append({'role': 'assistant', 'content': _txt})
        msgs.append({'role': 'user', 'content': [{'type': 'text', 'text': q}] + imgs})
        r = requests.post(base_url.rstrip('/') + '/chat/completions', headers={'Authorization': f"Bearer {key or 'none'}", 'Content-Type': 'application/json'}, json={'model': model, 'messages': msgs, 'max_tokens': max_tokens, 'temperature': temp}, timeout=240)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()

    def _bad(c, first):
        return not first or bool(_GEMMA_THINK.match(c)) or len(first) < 30
    c = _ask(0.2)
    first = c.splitlines()[0].strip() if c else ''
    for temp in (0.6, 0.9):
        if not _bad(c, first):
            return first
        c = _ask(temp)
        first = c.splitlines()[0].strip() if c else ''
    if not _bad(c, first):
        return first
    m = _GEMMA_SENT.search(c)
    return m.group(1).strip() if m else first

def caption_cosmos(reasoner_host, reasoner_port, media_path, prompt):
    import requests
    frames, is_video = media_frames(media_path, 1)
    r = requests.post(f'{reasoner_host}:{reasoner_port}/reason', json={'prompt': prompt, 'image_b64': frames[0] if frames else None, 'max_new_tokens': 300}, timeout=300)
    r.raise_for_status()
    j = r.json()
    return (j.get('answer') or j.get('text') or '').strip()
_VLM = {}

def caption_local_vlm(vlm_dir, frames_b64, prompt, max_tokens=160, device='cuda', four_bit=False):
    import torch, base64, io
    from PIL import Image
    if 'model' not in _VLM:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f'[curate] loading local VLM {vlm_dir} on {device} (4bit={four_bit})...', flush=True)
        kw = {}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
            kw['device_map'] = {'': device}
        else:
            kw['torch_dtype'] = torch.bfloat16
            kw['device_map'] = {'': device}
        _VLM['model'] = Qwen2_5_VLForConditionalGeneration.from_pretrained(vlm_dir, **kw).eval()
        _VLM['proc'] = AutoProcessor.from_pretrained(vlm_dir)
        _VLM['device'] = device
    model, proc, dev = (_VLM['model'], _VLM['proc'], _VLM['device'])
    imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB') for b in frames_b64[:1]]
    messages = [{'role': 'user', 'content': [{'type': 'image'} for _ in imgs] + [{'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=imgs, return_tensors='pt').to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

def _relevance(text: str) -> float:
    m = re.search('(\\d*\\.?\\d+)', text or '')
    if not m:
        return 0.0
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--media-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--goal', required=True, help='text description of what to learn from this dataset')
    ap.add_argument('--example', default='', help='inline example caption(s) for format/style')
    ap.add_argument('--example-file', default=None, help='file of example captions (one per line)')
    ap.add_argument('--captioner', choices=['openai', 'gemma', 'cosmos', 'local'], default='openai')
    ap.add_argument('--vlm-dir', default='', help='(--captioner local) path to a local Qwen2.5-VL dir')
    ap.add_argument('--vlm-device', default='cuda', help='(--captioner local) cuda or an explicit cuda:<ordinal> device')
    ap.add_argument('--vlm-4bit', action='store_true', help='(--captioner local) bnb NF4 load (fits the 3070)')
    ap.add_argument('--openai-base-url', default='')
    ap.add_argument('--openai-model', default='kimi-k2.6')
    ap.add_argument('--openai-key', default=os.environ.get('OPENAI_API_KEY', ''))
    ap.add_argument('--reasoner-host', default='<mac-ip>')
    ap.add_argument('--reasoner-port', type=int, default=9200)
    ap.add_argument('--frames', type=int, default=4, help='frames sampled per video for the VLM')
    ap.add_argument('--min-relevance', type=float, default=0.0, help='drop items below this (0 = keep all)')
    ap.add_argument('--limit', type=int, default=0, help='cap items processed (0 = all)')
    a = ap.parse_args()
    example = a.example
    if a.example_file and os.path.exists(a.example_file):
        example = open(a.example_file, encoding='utf-8').read().strip()
    files = sorted((p for p in glob.glob(os.path.join(a.media_dir, '**', '*'), recursive=True) if os.path.splitext(p)[1].lower() in IMG_EXT | VID_EXT))
    if a.limit:
        files = files[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    print(f'[curate] {len(files)} media files; captioner={a.captioner}; goal={a.goal!r}', flush=True)

    def cap(frames_b64, media_path, prompt):
        if a.captioner == 'openai':
            return caption_openai(a.openai_base_url, a.openai_model, a.openai_key, frames_b64, prompt)
        if a.captioner == 'gemma':
            return caption_gemma(a.openai_base_url, a.openai_model, a.openai_key, frames_b64, prompt)
        if a.captioner == 'local':
            return caption_local_vlm(a.vlm_dir, frames_b64, prompt, device=a.vlm_device, four_bit=a.vlm_4bit)
        return caption_cosmos(a.reasoner_host, a.reasoner_port, media_path, prompt)
    manifest, kept, dropped = ([], 0, 0)
    for i, f in enumerate(files):
        try:
            frames_b64, is_video = media_frames(f, a.frames)
            if not frames_b64:
                print(f'[curate] skip (no frames): {f}', flush=True)
                continue
            rel = 1.0
            if a.min_relevance > 0:
                rel = _relevance(cap(frames_b64, f, _prompt(a.goal, example, is_video, 'relevance')))
                if rel < a.min_relevance:
                    dropped += 1
                    print(f'[curate] drop rel={rel:.2f} {os.path.basename(f)}', flush=True)
                    continue
            caption = cap(frames_b64, f, _prompt(a.goal, example, is_video, 'caption'))
            stem = f"{kept:04d}_{re.sub('[^a-zA-Z0-9]+', '_', os.path.splitext(os.path.basename(f))[0])[:40]}"
            ext = os.path.splitext(f)[1].lower()
            shutil.copy(f, os.path.join(a.out, stem + ext))
            open(os.path.join(a.out, stem + '.txt'), 'w', encoding='utf-8').write(caption)
            manifest.append({'media': stem + ext, 'caption': caption, 'is_video': is_video, 'relevance': rel, 'source': f})
            kept += 1
            print(f'[curate] [{i + 1}/{len(files)}] kept rel={rel:.2f}: {caption[:80]}', flush=True)
        except Exception as e:
            print(f'[curate] ERROR on {f}: {type(e).__name__}: {str(e)[:160]}', flush=True)
    json.dump({'goal': a.goal, 'example': example, 'captioner': a.captioner, 'count': kept, 'items': manifest}, open(os.path.join(a.out, 'manifest.json'), 'w', encoding='utf-8'), indent=1)
    open(os.path.join(a.out, 'dataset.md'), 'w', encoding='utf-8').write(f'# Dataset: {a.goal}\n\n{kept} items (dropped {dropped} below relevance {a.min_relevance}). Captioned by {a.captioner}. Format: `<stem>.<ext>` media + `<stem>.txt` caption. Train: `python -m cosmos_windows.train_lora --dataset {a.out} ...`\n')
    print(f'[curate] DONE kept={kept} dropped={dropped} -> {a.out}', flush=True)
    print('RESULT', 'DONE' if kept > 0 else 'EMPTY', flush=True)
if __name__ == '__main__':
    main()
