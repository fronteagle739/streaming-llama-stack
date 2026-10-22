# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/minimax_h3_world.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import torch
import torch.nn.functional as F

def directed_groups(tokens, spans, video, device):
    annotations = {j for lo, hi in spans for j in range(lo, hi)}
    videos = set(video.reshape(-1).tolist())
    common = [j for j in range(tokens) if j not in annotations]
    static = [j for j in common if j not in videos]
    pairs = [(static, common)]
    for i, (lo, hi) in enumerate(spans):
        own = list(range(lo, hi))
        frame = video[i].tolist()
        pairs.extend([(frame, sorted(common + own)), (own, sorted(static + frame + own))])
    return [(torch.tensor(q, device=device, dtype=torch.long), torch.tensor(k, device=device, dtype=torch.long), torch.tensor([0, len(q)], device=device, dtype=torch.int32), torch.tensor([0, len(k)], device=device, dtype=torch.int32)) for q, k in pairs if q]

def directed_attention(q, k, v, groups, flash=None):
    output = torch.empty_like(q)
    for qi, ki, qb, kb in groups:
        qq, kk, vv = (q.index_select(0, qi), k.index_select(0, ki), v.index_select(0, ki))
        if flash is not None:
            result = flash(qq, kk, vv, qb, kb, len(qi), len(ki), dropout_p=0.0, causal=False)
            if isinstance(result, tuple):
                result = result[0]
        else:
            result = F.scaled_dot_product_attention(qq.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1)).transpose(0, 1)
        output.index_copy_(0, qi, result)
    return output

def bind_action_layout(layout, spans, geometry, device):
    text_len = len(layout['text_pos'])
    if len(spans) != geometry.video_t or not spans or spans[0][0] < 1:
        raise ValueError('H3-World requires one annotation per latent after a nonempty scene prompt')
    cursor = spans[0][0]
    for lo, hi in spans:
        if lo != cursor or not lo < hi <= text_len:
            raise ValueError('H3-World annotations must tile the text tail without gaps')
        cursor = hi
    if cursor != text_len:
        raise ValueError('H3-World annotation tail must end at text length')
    video = layout.get('img_output_pos', layout['img_pos']).reshape(geometry.video_t, -1)
    positions = layout['img_position_ids']
    times = positions[video[:, 0], 0] - positions[video[0, 0], 0]
    origin = text_len - float(times[-1]) - 1
    if origin < spans[0][0]:
        raise ValueError('H3-World annotations are too short for mirrored temporal positions')
    for i, (lo, hi) in enumerate(spans):
        positions[layout['text_pos'][lo:hi], 0] = origin + times[i]
    layout['action_refiner_bounds'] = [0, spans[0][0]] + [hi for lo, hi in spans]
    absolute = [(int(layout['text_pos'][lo]), int(layout['text_pos'][hi - 1]) + 1) for lo, hi in spans]
    layout['action_groups'] = directed_groups(layout['used_len'], absolute, video, device)

def encode_action_script(model_dir, head, script, cache, device, encode):
    preencode_actions(model_dir, script, cache, device, encode)
    rows, spans, cursor = ([head], [], len(head))
    for sentence in script:
        row = cache[sentence].to(device)
        rows.append(row)
        spans.append((cursor, cursor + len(row)))
        cursor += len(row)
    return (torch.cat(rows), spans)

def preencode_actions(model_dir, sentences, cache, device, encode, emit=lambda ev: None):
    wanted = list(dict.fromkeys(sentences))
    if len(wanted) > 256:
        raise ValueError('At most 256 distinct World actions can be encoded at once')
    done = 0
    for sentence in wanted:
        if sentence in cache:
            continue
        while len(cache) >= 256:
            del cache[next((key for key in cache if key not in wanted))]
        cache[sentence] = encode(model_dir, sentence, device=device).detach().cpu()
        done += 1
        emit(dict(event='world_action_preencoded', sentence=sentence, cached=len(cache)))
    return done

def yield_world_residency(session, drain, emit):
    import gc
    drain()
    had_model = 'model' in session
    streamer = session.pop('streamer', None)
    if streamer is not None:
        streamer.close()
    del streamer
    session.clear()
    gc.collect()
    torch.cuda.empty_cache()
    if had_model:
        emit(dict(event='world_residency_yielded', reason='text_encode_or_noise_retry'))

class H3PureNoiseError(RuntimeError):
    pass

def check_world_workspace(session, budget, device):
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)
    need = budget['vae_peak_bytes'] + budget['workspace_bytes'] + 500000000
    if 'model' not in session:
        need += budget['weight_bytes']
    if free < need:
        raise RuntimeError(f'World needs {need / 1000000000.0:.2f} GB free for resident weights/workspace, but the shared GPU now has {free / 1000000000.0:.2f} GB. No DiT work started.')

def pure_noise_metrics(frames):
    import numpy as np
    count = int(frames.shape[2])
    samples = [min(count - 1, max(0, count * n // 4)) for n in (1, 2, 3)]
    x = frames[0, :, samples].detach().float().cpu().numpy()
    if frames.dtype == torch.uint8:
        x /= 255.0
    elif float(np.nanmin(x)) < 0:
        x = (x + 1) * 0.5
    if not np.isfinite(x).all():
        return dict(pure_noise=True, nonfinite=True)
    votes, ratios, correlations = ([], [], [])
    for frame in x.transpose(1, 0, 2, 3):
        h, w = frame.shape[-2:]
        block = min(32, h // 4, w // 4)
        if block < 2:
            return dict(pure_noise=False, too_small=True)
        frame = frame[:, :h // block * block, :w // block * block]
        var = float(frame.var(axis=(1, 2)).mean())
        coarse = frame.reshape(3, h // block, block, w // block, block).mean(axis=(2, 4))
        ratio = float(coarse.var(axis=(1, 2)).mean()) / max(var, 1e-08)
        centered = frame.reshape(3, -1) - frame.mean(axis=(1, 2))[:, None]
        norm = np.sqrt((centered * centered).mean(axis=1))
        corr = centered @ centered.T / centered.shape[1] / np.maximum(norm[:, None] * norm[None, :], 1e-08)
        colour_corr = float(np.mean([corr[0, 1], corr[0, 2], corr[1, 2]]))
        votes.append(var > 0.008 and ratio < 0.12 and (colour_corr < 0.55))
        ratios.append(round(ratio, 5))
        correlations.append(round(colour_corr, 5))
    return dict(pure_noise=sum(votes) >= 2, coarse_ratio=ratios, colour_correlation=correlations)

def retry_world_noise(render, release, emit):
    for attempt in range(2):
        try:
            return render(attempt)
        except H3PureNoiseError:
            emit(dict(event='world_noise_retry' if attempt == 0 else 'world_noise_failed', attempt=attempt + 1))
            if attempt:
                raise
            release()

def encode_world_head(model_dir, prompt, image_path, width, height, device):
    from pathlib import Path
    from PIL import Image, ImageOps
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding
    from .minimax_h3_text_encoder import build_streamed_h3_qwen_encoder, release_h3_qwen_encoder
    root = Path(model_dir)
    processor = AutoProcessor.from_pretrained(root / 'processor', local_files_only=True)
    with Image.open(image_path) as source:
        picture = ImageOps.fit(source.convert('RGB'), (width, height), method=Image.Resampling.LANCZOS)
    vision = processor.image_processor(images=[picture], return_tensors='pt')
    count = int(vision['image_grid_thw'][0].prod()) // processor.image_processor.merge_size ** 2
    tokenizer = processor.tokenizer
    prefix = tokenizer('<Picture 1>: ', add_special_tokens=False)['input_ids']
    image_ids = [tokenizer.convert_tokens_to_ids('<|vision_start|>')]
    image_ids += [tokenizer.convert_tokens_to_ids('<|image_pad|>')] * count
    image_ids += [tokenizer.convert_tokens_to_ids('<|vision_end|>')]
    ids = prefix + image_ids + tokenizer(prompt, add_special_tokens=False)['input_ids']
    tags = [1] * len(prefix) + [0] * len(image_ids) + [1] * (len(ids) - len(prefix) - len(image_ids))
    model, streamer, _ = build_streamed_h3_qwen_encoder(root / 'text_encoder', device=device, include_vision=True)
    try:
        vision_model = model.model.visual
        head_dim = vision_model.config.hidden_size // vision_model.config.num_heads
        vision_model.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2).to(device)
        input_ids = torch.tensor([ids], device=device)
        with torch.inference_mode():
            hidden = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), mm_token_type_ids=(input_ids == tokenizer.convert_tokens_to_ids('<|image_pad|>')).long(), pixel_values=vision['pixel_values'].to(device, torch.bfloat16), image_grid_thw=vision['image_grid_thw'].to(device), use_cache=False, return_dict=True).last_hidden_state[0].to(torch.bfloat16)
        return (hidden, tags)
    finally:
        release_h3_qwen_encoder(model, streamer)
