# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/bpe_tokenizer_from_gguf.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import sys

def build_bpe_tokenizer_from_gguf(gguf_path_or_reader):
    from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast
    sys.path.insert(0, '')
    if isinstance(gguf_path_or_reader, str):
        from cosmos_windows.gguf_qat.streamed_reader import StreamedGGUF
        r = StreamedGGUF(gguf_path_or_reader, device='cpu')
        own = True
    else:
        r = gguf_path_or_reader
        own = False
    tokens = list(r.meta('tokenizer.ggml.tokens'))
    ttypes = list(r.meta('tokenizer.ggml.token_type') or [1] * len(tokens))
    merges_raw = list(r.meta('tokenizer.ggml.merges') or [])
    bos_id = r.meta('tokenizer.ggml.bos_token_id')
    eos_id = r.meta('tokenizer.ggml.eos_token_id')
    pad_id = r.meta('tokenizer.ggml.padding_token_id')
    add_bos = r.meta('tokenizer.ggml.add_bos_token')
    chat_template = r.meta('tokenizer.chat_template', None)
    if own:
        r.close()
    vocab = {tok: i for i, tok in enumerate(tokens)}
    merges = []
    for m in merges_raw:
        parts = m.split(' ')
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))
    model = BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False)
    tk = Tokenizer(model)
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    specials = []
    for i, (tok, tt) in enumerate(zip(tokens, ttypes)):
        if int(tt) in (3, 4):
            specials.append(tok)

    def name_of(idx):
        return tokens[int(idx)] if idx is not None and 0 <= int(idx) < len(tokens) else None
    bos_tok = name_of(bos_id)
    eos_tok = name_of(eos_id)
    pad_tok = name_of(pad_id)
    hf = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token=bos_tok, eos_token=eos_tok, pad_token=pad_tok, unk_token=None, additional_special_tokens=[s for s in specials if s not in (bos_tok, eos_tok, pad_tok)][:512])
    hf.add_bos_token = bool(add_bos) if add_bos is not None else False
    if chat_template:
        hf.chat_template = chat_template
    return hf
if __name__ == '__main__':
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb
    start_rss_guard(limit_gb=8.0, label='bpe_tok')
    which = sys.argv[1] if len(sys.argv) > 1 else 'step'
    if which == 'kimi':
        gguf = ''
    else:
        gguf = ''
    tok = build_bpe_tokenizer_from_gguf(gguf)
    print('type:', type(tok).__name__, 'vocab', tok.vocab_size, 'bos', tok.bos_token_id, 'eos', tok.eos_token_id, 'pad', tok.pad_token_id)
    for s in ['who are you?', 'MA name is JEFF!!', 'The capital of France is Paris.', 'Hello, world!']:
        ids = tok(s, add_special_tokens=False)['input_ids']
        back = tok.decode(ids)
        print(f"  {s!r:40s} -> {ids[:16]}{('...' if len(ids) > 16 else '')} -> {back!r}  RT={('OK' if back == s else 'DIFF')}")
    try:
        msgs = [{'role': 'user', 'content': 'who are you?'}]
        s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        print('chat-template str:', repr(s)[:240])
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        print('chat-template ids len', len(ids))
    except Exception as e:
        import traceback
        traceback.print_exc()
    print(f'peak RSS {peak_rss_gb():.3f} GB')
